"""
CognitiveEval: A package for running cognitive evaluation tests.

Includes:
- Wisconsin Card Sorting Test (WCST)
- Spatial Working Memory (SWM) test
- Raven's Progressive Matrices (RAPM)
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
    from .WCST.wcst import run_wcst, run_wcst_with_env  # type: ignore
except Exception:
    run_wcst = None  # type: ignore
    run_wcst_with_env = None  # type: ignore

try:
    from .SWM.main import swm_main, run_swm_with_env  # type: ignore
except Exception:
    swm_main = None  # type: ignore
    run_swm_with_env = None  # type: ignore

try:
    from .RAPM.rapm_evaluation import run_rapm_evaluation, run_rapm_with_env  # type: ignore
except Exception:
    run_rapm_evaluation = None  # type: ignore
    run_rapm_with_env = None  # type: ignore

# Environment classes
try:
    from .WCST.wcst_env import WCSTEnv  # type: ignore
    from .SWM.swm_env import SWMEnv  # type: ignore
    from .RAPM.rapm_env import RAPMEnv  # type: ignore
except Exception:
    WCSTEnv = None  # type: ignore
    SWMEnv = None  # type: ignore
    RAPMEnv = None  # type: ignore

# Base environment
try:
    from .shared.base_env import CognitiveEnv, StepResult, ActionStatus  # type: ignore
except Exception:
    CognitiveEnv = None  # type: ignore
    StepResult = None  # type: ignore
    ActionStatus = None  # type: ignore

__all__ = []
if run_wcst is not None:
    __all__.append("run_wcst")
if run_wcst_with_env is not None:
    __all__.append("run_wcst_with_env")
if swm_main is not None:
    __all__.append("swm_main")
if run_swm_with_env is not None:
    __all__.append("run_swm_with_env")
if run_rapm_evaluation is not None:
    __all__.append("run_rapm_evaluation")
if run_rapm_with_env is not None:
    __all__.append("run_rapm_with_env")
if score is not None:
    __all__.append("score")
# Environment classes
if WCSTEnv is not None:
    __all__.append("WCSTEnv")
if SWMEnv is not None:
    __all__.append("SWMEnv")
if RAPMEnv is not None:
    __all__.append("RAPMEnv")
if CognitiveEnv is not None:
    __all__.extend(["CognitiveEnv", "StepResult", "ActionStatus"])
