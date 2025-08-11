"""
CognitiveEval: A package for running cognitive evaluation tests.

Includes:
- Wisconsin Card Sorting Test (WCST)
- Spatial Working Memory (SWM) test
"""

__version__ = "1.0.0"
__author__ = "CognitiveEval Team"

# Export score (from legacy implementation) if available
try:
    from .main import score  # type: ignore
except Exception:

    def score(*_, **__):  # fallback no-op
        raise NotImplementedError("score function not available in this context")


# Primary public APIs
try:
    from .WCST.wcst import run_wcst  # type: ignore
except Exception:
    run_wcst = None  # type: ignore

try:
    from .SWM.main import swm_main  # type: ignore
except Exception:
    swm_main = None  # type: ignore

__all__ = []
if run_wcst is not None:
    __all__.append("run_wcst")
if swm_main is not None:
    __all__.append("swm_main")
if score is not None:
    __all__.append("score")
