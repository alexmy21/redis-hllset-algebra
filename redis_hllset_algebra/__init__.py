"""HLLSet Algebra – set-algebra operations for Redis HyperLogLog structures.

All heavy computation runs inside Redis as Lua scripts, keeping round-trips
to a minimum and results consistent across concurrent callers.

Exported symbols
----------------
HLLSetAlgebra   – the main client class
"""

from .hllset import HLLSetAlgebra

__all__ = ["HLLSetAlgebra"]
__version__ = "0.1.0"
