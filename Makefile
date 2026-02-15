SHELL := /bin/bash

BUILD_DIR ?= build
CMAKE ?= cmake
PYTHON ?= python3
PYTEST ?= pytest
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

.DEFAULT_GOAL := quick

.PHONY: help configure build test-cpp test-py test demo quick clean

help:
	@echo "Targets:"
	@echo "  make quick         # Configure, build, C++ tests, Python tests"
	@echo "  make build         # Configure + C++ build"
	@echo "  make test          # Run C++ and Python tests"
	@echo "  make demo          # Run Python demo (direct kernel path)"
	@echo "  make clean         # Remove build artifacts"

configure:
	$(CMAKE) -S . -B $(BUILD_DIR)

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

test-cpp: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

test-py: build
	PYTHONPATH=python $(PYTEST) -q python/tests

test: test-cpp test-py

demo: build
	PYTHONPATH=python $(PYTHON) python/examples/demo_pricing.py

quick: test

clean:
	$(CMAKE) -E rm -rf $(BUILD_DIR)
	$(CMAKE) -E rm -rf target
