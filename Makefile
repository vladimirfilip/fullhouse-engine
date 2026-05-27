.PHONY: install install-train install-train-gpu install-system install-vm \
        build-cpp clean-cpp train train-quick \
        demo test validate clean

# ── VM Python config ────────────────────────────────────────────────────────
# install-vm pins to Python 3.10 because eval7 0.1.7 ships pre-generated C
# files that #include "longintrepr.h" (a header Python 3.11 moved into
# Include/cpython/). Matches sandbox/Dockerfile.
VENV   ?= .venv
PY310  ?= python3.10
# When the venv exists, prefer it for train/build-cpp; otherwise system python3.
VPY    := $(if $(wildcard $(VENV)/bin/python),$(abspath $(VENV))/bin/python,python3)

# ── Python: sandbox-runtime deps (also fine for local dev) ───────────────────
install:
	@echo ">> Installing Cython<3 (eval7 build dep)"
	pip3 install "Cython<3"
	@echo ">> Installing eval7 with --no-build-isolation"
	pip3 install --no-build-isolation eval7==0.1.7
	@echo ">> Installing rest of sandbox requirements"
	pip3 install flask numpy scipy treys scikit-learn

# ── Python: extra deps for offline Deep CFR training (CPU wheel) ─────────────
install-train: install
	@echo ">> Installing training-only deps (PyTorch CPU, pybind11)"
	pip3 install --index-url https://download.pytorch.org/whl/cpu torch
	pip3 install pybind11

# ── Python: extra deps for offline Deep CFR training (CUDA 12.1 wheel) ───────
install-train-gpu: install
	@echo ">> Installing training-only deps (PyTorch CUDA 12.1, pybind11)"
	pip3 install --index-url https://download.pytorch.org/whl/cu121 torch
	pip3 install pybind11

# ── System packages required to build the C++ data-gen extension + 3.10 venv ─
# Eigen + pybind11 are fetched by CMake; we just need a toolchain + git +
# python3.10 (for the VM venv — see install-vm).
install-system:
	@echo ">> Installing system build deps (Debian/Ubuntu via apt-get)"
	@if command -v apt-get >/dev/null 2>&1; then \
	  sudo apt-get update && \
	  sudo apt-get install -y --no-install-recommends \
	    build-essential cmake git ca-certificates software-properties-common; \
	  if ! apt-cache show python3.10-venv >/dev/null 2>&1; then \
	    echo ">> python3.10 not in default repos — adding deadsnakes PPA"; \
	    sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update; \
	  fi; \
	  sudo apt-get install -y --no-install-recommends \
	    python3.10 python3.10-venv python3.10-dev; \
	else \
	  echo "!! apt-get not found. Install: build-essential, cmake, git,"; \
	  echo "   python3.10, python3.10-venv, python3.10-dev"; \
	  exit 1; \
	fi

# ── Create $(VENV) with Python 3.10 if it doesn't exist ──────────────────────
$(VENV)/bin/python:
	@command -v $(PY310) >/dev/null 2>&1 || { \
	  echo "!! $(PY310) not on PATH. Run 'make install-system' first"; \
	  echo "   (or install python3.10 + python3.10-venv manually)."; exit 1; }
	@echo ">> Creating venv at $(VENV) with $(PY310)"
	$(PY310) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip

# ── One-shot bootstrap for a fresh Linux training VM (GPU) ───────────────────
# Builds a self-contained Python 3.10 venv at $(VENV), installs all sandbox +
# training deps into it, then builds the C++ extension against that interpreter.
install-vm: install-system $(VENV)/bin/python
	@echo ">> Installing Cython<3 (eval7 build dep)"
	$(VENV)/bin/pip install "Cython<3"
	@echo ">> Installing eval7 with --no-build-isolation"
	$(VENV)/bin/pip install --no-build-isolation eval7==0.1.7
	@echo ">> Installing rest of sandbox requirements"
	$(VENV)/bin/pip install flask numpy scipy treys scikit-learn
	@echo ">> Installing training-only deps (PyTorch CUDA 12.1, pybind11)"
	$(VENV)/bin/pip install --index-url https://download.pytorch.org/whl/cu121 torch
	$(VENV)/bin/pip install pybind11
	@$(MAKE) build-cpp
	@echo ">> VM ready. Activate the venv:"
	@echo ">>   source $(VENV)/bin/activate"
	@echo ">> Then run 'make train-quick' for a smoke test."

# ── Build the C++ Deep CFR data-generation extension ─────────────────────────
# Prefers $(VENV)/bin/python when present so pybind11 binds against the 3.10
# venv on VMs; falls back to system python3 for local dev.
build-cpp:
	@echo ">> Configuring deep_cfr_cpp (Release) against $(VPY)"
	cmake -S bots/vlad/deep_cfr_cpp -B bots/vlad/deep_cfr_cpp/build \
	      -DCMAKE_BUILD_TYPE=Release \
	      -DPython3_EXECUTABLE=$(VPY)
	@echo ">> Building deep_cfr_gen extension"
	cmake --build bots/vlad/deep_cfr_cpp/build --config Release -j

clean-cpp:
	rm -rf bots/vlad/deep_cfr_cpp/build

# ── Training entry points (require: install-train + build-cpp) ───────────────
# Uses $(VENV) Python if present (VM workflow), else system python3 (local dev).
train:
	$(VPY) -m bots.vlad.deep_cfr.train

train-quick:
	$(VPY) -m bots.vlad.deep_cfr.train --quick

# ── Engine targets (frozen) ──────────────────────────────────────────────────
demo:
	python3 demo.py

test:
	python3 -m pytest tests/ -q

validate:
	@if [ -z "$(BOT)" ]; then echo "usage: make validate BOT=bots/mybot/bot.py"; exit 1; fi
	python3 sandbox/validator.py $(BOT)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
