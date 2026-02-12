"""Shared fixtures for QuantKernel tests."""

import pytest
from quantkernel import QuantKernel


@pytest.fixture(scope="session")
def qk():
    """Shared QuantKernel instance (loads library once)."""
    return QuantKernel()
