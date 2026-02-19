from __future__ import annotations

import os
from pathlib import Path

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Cython is required to build quantkernel._native_batch. "
        "Install it with: python -m pip install cython"
    ) from exc

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
PY_ROOT = Path(__file__).resolve().parent

default_lib_dirs = [ROOT / "build" / "cpp", ROOT / "build", ROOT]
env_lib_dir = os.environ.get("QK_LIB_PATH")
lib_dirs = [Path(env_lib_dir)] if env_lib_dir else default_lib_dirs
lib_dirs = [str(p) for p in lib_dirs if p.exists()]
if not lib_dirs:
    lib_dirs = [str(ROOT / "build"), str(ROOT / "build" / "cpp")]

extra_link_args = []
if os.name != "nt":
    for d in lib_dirs:
        extra_link_args.extend(["-Wl,-rpath," + d])

extensions = [
    Extension(
        "quantkernel._native_batch",
        [str(PY_ROOT / "quantkernel" / "_native_batch.pyx")],
        include_dirs=[np.get_include(), str(ROOT / "cpp" / "include")],
        library_dirs=lib_dirs,
        libraries=["quantkernel"],
        language="c++",
        extra_link_args=extra_link_args,
    )
]

setup(
    name="quantkernel-native-batch",
    version="0.0.1",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
