SHELL := /bin/bash

BUILD_DIR ?= build
CMAKE ?= cmake
PYTHON ?= python3
PYTEST ?= pytest
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

.DEFAULT_GOAL := quick

.PHONY: help configure build build-wheel bench test-cpp test-py test demo quick clean clean-fuzz clean-caches clean-all clean-venv

help:
	@echo "Targets:"
	@echo "  make quick         # Configure, build, C++ tests, Python tests"
	@echo "  make build         # Configure + C++ build"
	@echo "  make build-wheel   # Build Python wheel (platform-specific)"
	@echo "  make bench         # Run scalar/batch benchmark table script"
	@echo "  make test          # Run C++ and Python tests"
	@echo "  make demo          # Run Python demo (direct kernel path)"
	@echo "  make clean         # Remove build artifacts"
	@echo "  make clean-fuzz    # Remove fuzztest build artifacts"
	@echo "  make clean-caches  # Remove local caches (pycache/pytest/compile_commands)"
	@echo "  make clean-all     # clean + clean-fuzz + clean-caches"
	@echo "  make clean-venv    # Remove local .venv"

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

build-wheel:
	$(PYTHON) -m pip wheel . --no-deps -w dist

bench: build
	PYTHONPATH=python QK_LIB_PATH="$(CURDIR)/$(BUILD_DIR)/cpp" $(PYTHON) examples/benchmark_scalar_batch_cpp.py --n 50000 --repeats 3

test-cpp: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

test-py: build
	PYTHONPATH=python QK_LIB_PATH="$(CURDIR)/$(BUILD_DIR)/cpp" $(PYTEST) -q python/tests

test: test-cpp test-py

demo: build
	PYTHONPATH=python $(PYTHON) examples/demo_pricing.py

quick: test

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)
	$(CMAKE) -E rm -rf target

clean-fuzz:
	$(CMAKE) -E rm -rf fuzztest/build
	$(CMAKE) -E rm -rf fuzztest/dist

clean-caches:
	$(CMAKE) -E rm -rf .pytest_cache
	find python -type d -name __pycache__ -prune -exec rm -rf {} +
	$(CMAKE) -E rm -f compile_commands.json

clean-all: clean clean-fuzz clean-caches

clean-venv:
	$(CMAKE) -E rm -rf .venv
