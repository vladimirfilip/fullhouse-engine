"""
Shared-memory open-addressing hash table for parallel preflop CFR training.

Four SharedMemory blocks store info-set data for all workers without any
pickling or IPC per iteration:

    keys     int64   [capacity]       — INT64_MIN marks an empty slot
    regrets  float64 [capacity, 9]    — RM+ cumulative regrets (≥0)
    strategy float32 [capacity, 9]    — iteration-weighted strategy sum
    visits   float64 [capacity]       — true visit count (for prune)

Regret/strategy/visit updates are written directly (Hogwild) — ES-MCCFR
already produces noisy gradient estimates, so the rare lost update from
concurrent float64 writes on x86-64 is negligible.  New-slot insertion uses
a single process-wide lock (amortised O(1) after the tree warms up).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
from multiprocessing.shared_memory import SharedMemory

import numpy as np

_SENTINEL = np.iinfo(np.int64).min   # -2^63: marks an empty hash slot


class SharedHashTable:
    """
    Open-addressing flat hash table in shared memory.

    Typical usage
    -------------
    Main process::

        lock  = mp.Lock()
        table = SharedHashTable.create(capacity, lock)

    Worker initializer::

        table = SharedHashTable.attach(table.name_prefix, capacity, lock)

    Both ends receive numpy views into the same physical pages; worker writes
    are immediately visible to all other processes.
    """

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def create(cls, capacity: int, insert_lock: "mp.Lock") -> "SharedHashTable":
        """Allocate four SharedMemory blocks and return a zeroed table."""
        self = cls.__new__(cls)
        self.capacity     = capacity
        self._insert_lock = insert_lock
        self._owner       = True
        # Unique prefix: PID + 8 random hex digits avoids collisions between
        # concurrent runs on the same machine.
        self.name_prefix  = f"pfc{os.getpid():05d}{random.getrandbits(32):08x}"
        self._open(create=True)
        self.keys[:]     = _SENTINEL
        self.regrets[:]  = 0.0
        self.strategy[:] = 0.0
        self.visits[:]   = 0.0
        return self

    @classmethod
    def attach(cls, name_prefix: str, capacity: int,
               insert_lock: "mp.Lock") -> "SharedHashTable":
        """Attach to blocks created by the main process (worker-side)."""
        self = cls.__new__(cls)
        self.capacity     = capacity
        self.name_prefix  = name_prefix
        self._insert_lock = insert_lock
        self._owner       = False
        self._open(create=False)
        return self

    def _open(self, create: bool) -> None:
        cap = self.capacity
        pfx = self.name_prefix
        kw  = dict(create=create)
        self._shm_k = SharedMemory(name=f"{pfx}k", **kw, size=cap * 8)
        self._shm_r = SharedMemory(name=f"{pfx}r", **kw, size=cap * 9 * 8)
        self._shm_s = SharedMemory(name=f"{pfx}s", **kw, size=cap * 9 * 4)
        self._shm_v = SharedMemory(name=f"{pfx}v", **kw, size=cap * 8)
        self.keys     = np.ndarray(cap,      dtype=np.int64,   buffer=self._shm_k.buf)
        self.regrets  = np.ndarray((cap, 9), dtype=np.float64, buffer=self._shm_r.buf)
        self.strategy = np.ndarray((cap, 9), dtype=np.float32, buffer=self._shm_s.buf)
        self.visits   = np.ndarray(cap,      dtype=np.float64, buffer=self._shm_v.buf)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Detach numpy views and close SharedMemory handles (no unlink)."""
        del self.keys, self.regrets, self.strategy, self.visits
        for shm in (self._shm_k, self._shm_r, self._shm_s, self._shm_v):
            shm.close()

    def unlink(self) -> None:
        """Release OS shared-memory resources; call only from the creator."""
        if not self._owner:
            return
        for shm in (self._shm_k, self._shm_r, self._shm_s, self._shm_v):
            try:
                shm.unlink()
            except Exception:
                pass

    # ── Hash table ────────────────────────────────────────────────────────────

    def find_or_insert(self, key: int) -> int:
        """
        Return the slot index for `key`, allocating a new slot on first sight.

        Lock-free fast path: linear-probe until the key is found or a sentinel
        (empty) slot is reached.  If a sentinel is found, acquire the lock and
        re-probe from the hash origin to safely handle concurrent insertions.
        The lock is only held during the rare new-key path.
        """
        keys  = self.keys
        cap   = self.capacity
        start = slot = int(key % cap)

        # Lock-free scan.
        while True:
            k = int(keys[slot])
            if k == key:
                return slot
            if k == _SENTINEL:
                break
            slot = (slot + 1) % cap
            if slot == start:
                raise RuntimeError(
                    f"SharedHashTable full (capacity={cap:,}). "
                    "Increase SHARED_TABLE_CAPACITY in config.py.")

        # Potential new slot: re-probe under lock to avoid duplicate keys.
        with self._insert_lock:
            slot = start
            for _ in range(cap):
                k = int(keys[slot])
                if k == key:
                    return slot        # another worker already inserted it
                if k == _SENTINEL:
                    keys[slot] = key   # we win the race — claim this slot
                    return slot
                slot = (slot + 1) % cap
            raise RuntimeError(
                f"SharedHashTable full (capacity={cap:,}). "
                "Increase SHARED_TABLE_CAPACITY in config.py.")

    def n_info_sets(self) -> int:
        """Number of occupied slots (vectorised numpy scan)."""
        return int((self.keys != _SENTINEL).sum())

    # ── Checkpoint / restore ──────────────────────────────────────────────────

    def to_dicts(self) -> tuple[dict, dict, dict]:
        """
        Copy live shared data into Python dicts for checkpointing or export.
        Not thread-safe — call only when workers are paused or finished.
        """
        occupied = np.flatnonzero(self.keys != _SENTINEL)
        regret_sum:   dict[int, np.ndarray] = {}
        strategy_sum: dict[int, np.ndarray] = {}
        visit_sum:    dict[int, float]      = {}
        for slot in occupied:
            k = int(self.keys[slot])
            regret_sum[k]   = self.regrets[slot].copy()
            strategy_sum[k] = self.strategy[slot].astype(np.float64)
            visit_sum[k]    = float(self.visits[slot])
        return regret_sum, strategy_sum, visit_sum

    def from_dicts(self, regret_sum: dict, strategy_sum: dict,
                   visit_sum: dict) -> None:
        """Bulk-load checkpoint dicts into shared memory (before workers start)."""
        for k, rv in regret_sum.items():
            idx = self.find_or_insert(k)
            self.regrets[idx] = rv
            sv = strategy_sum.get(k)
            if sv is not None:
                self.strategy[idx] = sv.astype(np.float32)
            vv = visit_sum.get(k)
            if vv is not None:
                self.visits[idx] = float(vv)
