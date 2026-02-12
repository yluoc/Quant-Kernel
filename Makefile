SHELL := /bin/bash

BUILD_DIR ?= build
CMAKE ?= cmake
PYTHON ?= python3
PYTEST ?= pytest
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

.DEFAULT_GOAL := quick

.PHONY: help configure build runtime test-cpp test-py test demo demo-runtime quick clean

help:
	@echo "Targets:"
	@echo "  make quick         # Configure, build, runtime build, C++ tests, Python tests"
	@echo "  make build         # Configure + C++ build"
	@echo "  make runtime       # Build Rust runtime target via CMake"
	@echo "  make test          # Run C++ and Python tests"
	@echo "  make demo          # Run Python demo (direct kernel path)"
	@echo "  make demo-runtime  # Run Python demo through Rust runtime path"
	@echo "  make clean         # Remove build and Rust target artifacts"

configure:
	$(CMAKE) -S . -B $(BUILD_DIR)

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

runtime: build
	$(CMAKE) --build $(BUILD_DIR) --target quantkernel_runtime

test-cpp: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

test-py: runtime
	PYTHONPATH=python $(PYTEST) -q python/tests

test: test-cpp test-py

demo: runtime
	PYTHONPATH=python $(PYTHON) python/examples/demo_pricing.py

demo-runtime: runtime
	QK_USE_RUNTIME=1 \
	QK_PLUGIN_PATH=$(PWD)/$(BUILD_DIR)/cpp/libquantkernel.so \
	PYTHONPATH=python $(PYTHON) python/examples/demo_pricing.py

quick: test

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)
	$(CMAKE) -E rm -rf target
	$(CMAKE) -E rm -rf rust/runtime/target
	$(CMAKE) -E rm -rf rust/runtime/.tmp
