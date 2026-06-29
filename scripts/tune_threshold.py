import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from rpgp_demo.threshold import find_best_threshold
from rpgp_demo.torch_data import encode_text
from rpgp_demo.torch_model import build_relation_classifier


DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval_samples.json"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "relpilot_relation.pt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Tune relation prediction threshold.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Evaluation dataset JSON path")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model checkpoint path")
    return parser.parse_args(argv)


def _gold_relations(sample):
    return {triple["relation"] for triple in sample.get("triples", [])}


def main(argv=None):
    args = parse_args(argv)
    with open(args.dataset, "r", encoding="utf-8") as file:
        samples = json.load(file)

    checkpoint = torch.load(args.model, map_location="cpu")
    model = build_relation_classifier(
        checkpoint.get("model_type", "mean"),
        len(checkpoint["vocab"]),
        len(checkpoint["relations"]),
        checkpoint["embedding_dim"],
        checkpoint.get("hidden_channels", 48),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    probabilities = []
    with torch.no_grad():
        for sample in samples:
            ids = encode_text(sample["text"], checkpoint["vocab"], checkpoint["max_length"])
            tensor = torch.tensor([ids], dtype=torch.long)
            values = torch.sigmoid(model(tensor))[0].tolist()
            probabilities.append(dict(zip(checkpoint["relations"], values)))

    candidates = [round(value / 100, 2) for value in range(20, 86, 5)]
    result = find_best_threshold(probabilities, [_gold_relations(sample) for sample in samples], candidates)
    print(
        "Best threshold: "
        f"{result['threshold']:.2f}, "
        f"P={result['precision']:.4f}, "
        f"R={result['recall']:.4f}, "
        f"F1={result['f1']:.4f}"
    )
    return result


if __name__ == "__main__":
    main()
