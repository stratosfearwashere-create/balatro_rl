"""Compile the package's extensions with Cython, in place.

    python setup_cython.py build          # the typed extensions (*.pyx): scorer, cards, clone, solver
    python setup_cython.py clean          # remove the compiled extensions and generated C files
    python setup_cython.py build --pure   # also compile the pure-Python modules in MODULES as they are
    python setup_cython.py build --only balatro_rl/az/_solver.pyx

The typed extensions are where the speed is; each has a Python fallback (BALATRO_PURE=1 forces it) and the
same results bit for bit (tests/test_regression.py, tests/fidelity/test_fastscore.py). `--pure` compiles the
listed .py modules unchanged: measured at about 3% and it can change numeric types where Cython types an
expression of len() and `/` as C (an effect recorded as the int 3 becomes 3.0), which breaks the golden
regression, so it is off by default. A compiled module (.pyd / .so) shadows its .py file: after editing a
compiled module rebuild or `clean`, or the old code keeps running. Modules with a command line are never
compiled (`python -m` cannot run an extension module).
"""
from __future__ import annotations

import glob
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, "balatro_rl")

# pure-Python modules compiled as they are (no command line, imported by the hot paths)
MODULES = [
    "balatro_rl/sim/cards.py", "balatro_rl/sim/hands.py", "balatro_rl/sim/scoring.py", "balatro_rl/sim/jokers.py",
    "balatro_rl/sim/items.py", "balatro_rl/sim/game.py", "balatro_rl/sim/fastscore.py",
    "balatro_rl/az/actions.py", "balatro_rl/az/agent.py", "balatro_rl/az/features.py", "balatro_rl/az/search.py",
    "balatro_rl/az/shop.py", "balatro_rl/az/world.py",
    "balatro_rl/rewards/config.py", "balatro_rl/rewards/potential.py", "balatro_rl/rewards/strength.py",
    "balatro_rl/rewards/targets.py", "balatro_rl/rewards/novelty.py", "balatro_rl/rewards/growth.py",
    "balatro_rl/rewards/diagnostics.py",
    "balatro_rl/env.py", "balatro_rl/heuristic.py",
]
# typed rewrites
PYX = sorted(glob.glob(os.path.join(PKG, "**", "*.pyx"), recursive=True))

# Safe defaults only: the .py modules are compiled as Python, so negative indexes, bounds errors and None
# checks must behave exactly as in CPython (Cython infers `list` for locals, and boundscheck=False /
# wraparound=False would turn `lst[-1]` into an unchecked C index). The .pyx files set their own directives.
DIRECTIVES = {
    "language_level": 3,
    "binding": True,            # compiled functions keep signatures / introspection (dataclasses, pickling)
    "embedsignature": False,
}


def _targets(only: str | None, pure: bool = False):
    files = ([os.path.join(ROOT, m) for m in MODULES] if pure else []) + PYX
    if only:
        want = os.path.abspath(only)
        files = [f for f in files if os.path.abspath(f) == want]
        if not files:
            raise SystemExit(f"{only} is not in the module list")
    return files


def clean():
    removed = 0
    for pat in ("**/*.pyd", "**/*.so", "**/*.c", "**/*.html"):
        for f in glob.glob(os.path.join(PKG, pat), recursive=True):
            os.remove(f)
            removed += 1
    for d in (os.path.join(ROOT, "build"),):
        if os.path.isdir(d):
            shutil.rmtree(d)
    print(f"removed {removed} generated files")


CPP = "balatro_rl/sim/cpp/core.cpp"       # the C++ game core (pybind11), built after the Cython extensions


def _cpp_extension():
    """The C++ core (balatro_rl.sim._core), or None when pybind11 is not installed. It reads the compiled
    card's struct through the header the Cython build of _cards.pyx generates."""
    from setuptools import Extension
    try:
        import pybind11
    except ImportError:
        print("pybind11 is not installed: the C++ core is not built (pip install pybind11)")
        return None
    extra = ["-std=c++17", "-O2"] if sys.platform != "win32" else ["/std:c++17", "/O2", "/EHsc"]
    return Extension("balatro_rl.sim._core", [CPP], include_dirs=[pybind11.get_include(), "balatro_rl/sim",
                                                                 "balatro_rl/sim/cpp"],
                     language="c++", extra_compile_args=extra, depends=glob.glob("balatro_rl/sim/cpp/*.hpp"))


def build(only: str | None, jobs: int, force: bool = False, pure: bool = False):
    from Cython.Build import cythonize
    from setuptools import setup
    from setuptools import Extension
    files = _targets(only, pure) if only != CPP else []
    exts = []
    for f in files:
        rel = os.path.relpath(f, ROOT)
        name = rel[:-len(os.path.splitext(rel)[1])].replace(os.sep, ".")
        exts.append(Extension(name, [rel]))
    sys.argv = [sys.argv[0], "build_ext", "--inplace", f"--parallel={jobs}"]
    cy = cythonize(exts, compiler_directives=DIRECTIVES, nthreads=jobs, quiet=True, force=force, annotate=False)
    if exts:
        setup(name="balatro_rl_compiled", ext_modules=cy, zip_safe=False)
    if only is None or only == CPP:
        cpp = _cpp_extension()
        if cpp is not None and os.path.exists(os.path.join(ROOT, "balatro_rl/sim/_cards_api.h")):
            setup(name="balatro_rl_core", ext_modules=[cpp], zip_safe=False)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "clean"])
    p.add_argument("--only", default=None, help="one .pyx file, or balatro_rl/sim/cpp/core.cpp for the C++ core")
    p.add_argument("--force", action="store_true", help="regenerate every C file (after changing directives)")
    p.add_argument("--pure", action="store_true", help="also compile the pure-Python modules in MODULES")
    p.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    a = p.parse_args()
    os.chdir(ROOT)
    if a.cmd == "clean":
        clean()
    else:
        build(a.only, a.jobs, a.force, a.pure)


if __name__ == "__main__":
    main()
