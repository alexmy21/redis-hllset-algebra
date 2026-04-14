#!/usr/bin/env python
"""
Build script for Cython extensions.

Usage:
    python build_ext.py           # Build all extensions
    python build_ext.py --clean   # Clean and rebuild
"""

import os
import sys
import subprocess
from pathlib import Path

CORE_DIR = Path(__file__).parent / "core"

# Cython source files to compile
CYTHON_MODULES = [
    "hll_core.pyx",
    "bitvector_core.pyx",
]


def build_extensions():
    """Build all Cython extensions."""
    import numpy as np
    from Cython.Build import cythonize
    from setuptools import Extension
    from setuptools.command.build_ext import build_ext
    from setuptools.dist import Distribution
    
    extensions = []
    for module in CYTHON_MODULES:
        pyx_path = CORE_DIR / module
        if pyx_path.exists():
            name = f"core.{module.replace('.pyx', '')}"
            ext = Extension(
                name,
                sources=[str(pyx_path)],
                include_dirs=[np.get_include()],
                extra_compile_args=["-O3"],
            )
            extensions.append(ext)
            print(f"  Adding: {name}")
    
    if not extensions:
        print("No Cython modules found!")
        return
    
    # Cythonize
    print("\nCythonizing...")
    extensions = cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
        },
    )
    
    # Build
    print("\nBuilding...")
    dist = Distribution({"ext_modules": extensions})
    cmd = build_ext(dist)
    cmd.inplace = True
    cmd.ensure_finalized()
    cmd.run()
    
    print("\n✓ Build complete!")


def clean():
    """Remove compiled extensions and C files."""
    import glob
    
    patterns = [
        str(CORE_DIR / "*.so"),
        str(CORE_DIR / "*.c"),
        str(CORE_DIR / "*.html"),
    ]
    
    for pattern in patterns:
        for f in glob.glob(pattern):
            print(f"  Removing: {f}")
            os.remove(f)
    
    print("✓ Clean complete!")


if __name__ == "__main__":
    print(f"Working directory: {Path.cwd()}")
    print(f"Core directory: {CORE_DIR}")
    
    if "--clean" in sys.argv:
        print("\nCleaning...")
        clean()
    
    print("\nBuilding Cython extensions...")
    build_extensions()
