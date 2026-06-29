import argparse
import json
import sys
from pathlib import Path


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "eval_samples.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rpgp_demo.evaluate import evaluate_samples


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the RPGP demo extractor.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to eval_samples.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.dataset, "r", encoding="utf-8") as file:
        samples = json.load(file)
    metrics = evaluate_samples(samples)
    print("RPGP Demo Evaluation")
    print(f"Samples: {metrics['sample_count']}")
    print(f"Gold triples: {metrics['gold_count']}")
    print(f"Predicted triples: {metrics['predicted_count']}")
    print(f"Correct triples: {metrics['correct_count']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
