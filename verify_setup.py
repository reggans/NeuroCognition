#!/usr/bin/env python3
"""
Setup verification script for verifiers integration.
Checks all dependencies, rubrics, and environment configuration.
"""

import sys
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_import(module_name: str, description: str = "") -> bool:
    """Check if a module can be imported."""
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"   ✓ {description}: {module_name} (v{version})")
        return True
    except ImportError as e:
        print(f"   ✗ {description}: {module_name}")
        print(f"      Error: {e}")
        return False


def check_file(path: Path, description: str = "") -> bool:
    """Check if a file exists."""
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"   {status} {description}: {path.name}")
    return exists


def main():
    print("=" * 70)
    print("Verifiers Integration Setup Verification")
    print("=" * 70)

    # Check Python version
    print("\n1. Python Environment:")
    print(f"   Python: {sys.version}")
    python_ok = sys.version_info >= (3, 10)
    print(f"   {'✓' if python_ok else '✗'} Python >= 3.10 required")

    # Check core dependencies
    print("\n2. Core Dependencies:")
    deps = [
        ("torch", "PyTorch"),
        ("transformers", "Hugging Face Transformers"),
        ("datasets", "Hugging Face Datasets"),
        ("verifiers", "Verifiers Library"),
    ]

    all_imports_ok = True
    for module, desc in deps:
        if not check_import(module, desc):
            all_imports_ok = False

    # Check task-specific files
    print("\n3. Task Modules:")
    task_files = [
        (PROJECT_ROOT / "RAPM" / "rapm_verifiers_env.py", "RAPM Verifiers Env"),
        (PROJECT_ROOT / "SWM" / "swm_verifiers_env.py", "SWM Verifiers Env"),
        (PROJECT_ROOT / "WCST" / "wcst_verifiers_env.py", "WCST Verifiers Env"),
    ]

    files_ok = True
    for file_path, desc in task_files:
        if not check_file(file_path, desc):
            files_ok = False

    # Check rubrics
    print("\n4. Rubric Files:")
    rubric_files = [
        (PROJECT_ROOT / "RAPM" / "rapm_rubric.py", "RAPM Rubric"),
        (PROJECT_ROOT / "SWM" / "swm_rubric.py", "SWM Rubric"),
        (PROJECT_ROOT / "WCST" / "wcst_rubric.py", "WCST Rubric"),
    ]

    rubrics_ok = True
    for file_path, desc in rubric_files:
        if not check_file(file_path, desc):
            rubrics_ok = False

    # Test rubric imports and structure
    print("\n5. Rubric Class Structure:")
    rubric_tests = []
    try:
        from RAPM.rapm_rubric import RAPMRubric
        rubric = RAPMRubric(mode="image", answer_mode="mc")
        turn_funcs = rubric.turn_reward_funcs
        outcome_funcs = rubric.outcome_reward_funcs
        print(f"   ✓ RAPM Rubric: {len(turn_funcs)} turn funcs, {len(outcome_funcs)} outcome funcs")
        rubric_tests.append(True)
    except Exception as e:
        print(f"   ✗ RAPM Rubric: {e}")
        rubric_tests.append(False)

    try:
        from SWM.swm_rubric import SWMRubric
        rubric = SWMRubric()
        turn_funcs = rubric.turn_reward_funcs
        outcome_funcs = rubric.outcome_reward_funcs
        print(f"   ✓ SWM Rubric: {len(turn_funcs)} turn funcs, {len(outcome_funcs)} outcome funcs")
        rubric_tests.append(True)
    except Exception as e:
        print(f"   ✗ SWM Rubric: {e}")
        rubric_tests.append(False)

    try:
        from WCST.wcst_rubric import WCSTRubric
        rubric = WCSTRubric()
        turn_funcs = rubric.turn_reward_funcs
        outcome_funcs = rubric.outcome_reward_funcs
        print(f"   ✓ WCST Rubric: {len(turn_funcs)} turn funcs, {len(outcome_funcs)} outcome funcs")
        rubric_tests.append(True)
    except Exception as e:
        print(f"   ✗ WCST Rubric: {e}")
        rubric_tests.append(False)

    rubrics_class_ok = all(rubric_tests)

    # Check RAPM validators
    print("\n6. RAPM Validators:")
    rapm_validators = [
        (PROJECT_ROOT / "RAPM" / "text_rapm" / "validator.py", "RAPM Text Validator"),
        (
            PROJECT_ROOT / "RAPM" / "text_rapm" / "per_cell_constraints.py",
            "RAPM Constraints",
        ),
    ]

    validators_ok = True
    for file_path, desc in rapm_validators:
        if not check_file(file_path, desc):
            validators_ok = False

    # Check WCST utilities
    print("\n7. WCST Utilities:")
    wcst_files = [
        (PROJECT_ROOT / "WCST" / "utils.py", "WCST Generators"),
    ]

    wcst_ok = True
    for file_path, desc in wcst_files:
        if not check_file(file_path, desc):
            wcst_ok = False

    # Check GPU
    print("\n8. GPU Information:")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"   ✓ CUDA Available: True")
            print(f"     Device: {torch.cuda.get_device_name(0)}")
            print(
                f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )
        else:
            print(f"   ✗ CUDA Available: False (will use CPU)")
    except Exception as e:
        print(f"   ✗ Error checking GPU: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("Setup Summary:")
    print("=" * 70)

    all_ok = (
        python_ok
        and all_imports_ok
        and files_ok
        and rubrics_ok
        and rubrics_class_ok
        and validators_ok
        and wcst_ok
    )

    if all_ok:
        print("✓ All checks passed! Ready to test verifiers and train models.")
        print("\nNext steps:")
        print("1. Test rubrics and envs: python test_verifiers_env.py --task all")
        print("2. Train individual task: python wcst_mt_grpo_train.py")
        print("3. Train all tasks: python multi_task_mt_grpo_train.py")
        return 0
    else:
        print("✗ Some checks failed. Please resolve issues above.")
        print("\nMissing dependencies? Run:")
        print("  cd /root/Multi-Turn-RL-Agent && pip install -e .")
        return 1


if __name__ == "__main__":
    sys.exit(main())
