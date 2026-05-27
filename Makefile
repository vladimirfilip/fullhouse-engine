.PHONY: install install-train install-train-gpu install-system install-vm \
        build-cpp clean-cpp train train-quick \
        demo test validate clean

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

# ── System packages required to build the C++ data-gen extension ─────────────
# Eigen + pybind11 are fetched by CMake; we just need a toolchain + git.
install-system:
	@echo ">> Installing system build deps (Debian/Ubuntu via apt-get)"
	@if command -v apt-get >/dev/null 2>&1; then \
	  sudo apt-get update && \
	  sudo apt-get install -y --no-install-recommends \
	    build-essential cmake git python3-dev ca-certificates; \
	else \
	  echo "!! apt-get not found. Install: build-essential, cmake, git, python3-dev"; \
	  exit 1; \
	fi

# ── One-shot bootstrap for a fresh Linux training VM (GPU) ───────────────────
install-vm: install-system install-train-gpu build-cpp
	@echo ">> VM ready. Run 'make train-quick' for a smoke test."

# ── Build the C++ Deep CFR data-generation extension ─────────────────────────
build-cpp:
	@echo ">> Configuring deep_cfr_cpp (Release)"
	cmake -S bots/vlad/deep_cfr_cpp -B bots/vlad/deep_cfr_cpp/build \
	      -DCMAKE_BUILD_TYPE=Release
	@echo ">> Building deep_cfr_gen extension"
	cmake --build bots/vlad/deep_cfr_cpp/build --config Release -j

clean-cpp:
	rm -rf bots/vlad/deep_cfr_cpp/build

# ── Training entry points (require: install-train + build-cpp) ───────────────
train:
	python3 -m bots.vlad.deep_cfr.train

train-quick:
	python3 -m bots.vlad.deep_cfr.train --quick

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
