import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rpgp_demo.evaluate import evaluate_samples


def run_checks(include_tests=True):
    if include_tests:
        test_result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        tests = {
            "ok": test_result.returncode == 0,
            "returncode": test_result.returncode,
        }
    else:
        tests = {
            "ok": True,
            "returncode": 0,
        }
    with open(PROJECT_ROOT / "data" / "eval_samples.json", "r", encoding="utf-8") as file:
        metrics = evaluate_samples(json.load(file))
    return {
        "tests": tests,
        "evaluation": {
            "ok": metrics["f1"] >= 0.8,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "sample_count": metrics["sample_count"],
        },
    }


def main():
    results = run_checks()
    print("RelPilot Demo Check")
    print(f"Tests: {'OK' if results['tests']['ok'] else 'FAIL'}")
    print(
        "Evaluation: "
        f"P={results['evaluation']['precision']:.4f}, "
        f"R={results['evaluation']['recall']:.4f}, "
        f"F1={results['evaluation']['f1']:.4f}, "
        f"Samples={results['evaluation']['sample_count']}"
    )
    if not results["tests"]["ok"] or not results["evaluation"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
