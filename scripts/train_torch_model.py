import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rpgp_demo.data import RELATION_KEYWORDS
from rpgp_demo.evaluate import evaluate_samples
from rpgp_demo.models.torch_data import build_relation_labels, build_vocab, encode_text
from rpgp_demo.models.torch_model import build_relation_classifier


DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval_samples.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "relpilot_relation.pt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the lightweight RelPilot relation predictor.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Training dataset JSON path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Model checkpoint path")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-length", type=int, default=64, help="Max character length")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden-channels", type=int, default=48, help="TextCNN hidden channels")
    parser.add_argument("--model-type", choices=("mean", "textcnn"), default="textcnn", help="Classifier type")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    return parser.parse_args(argv)


def train(argv=None):
    args = parse_args(argv)
    torch.manual_seed(7)
    with open(args.dataset, "r", encoding="utf-8") as file:
        samples = json.load(file)

    relations = list(RELATION_KEYWORDS)
    vocab = build_vocab(sample["text"] for sample in samples)
    features = [
        encode_text(sample["text"], vocab, args.max_length)
        for sample in samples
    ]
    labels = [
        [build_relation_labels(sample, relations)[relation] for relation in relations]
        for sample in samples
    ]
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = build_relation_classifier(
        args.model_type,
        len(vocab),
        len(relations),
        args.embedding_dim,
        args.hidden_channels,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for _epoch in range(args.epochs):
        model.train()
        for input_ids, target in loader:
            optimizer.zero_grad()
            loss = criterion(model(input_ids), target)
            loss.backward()
            optimizer.step()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "relations": relations,
            "vocab": vocab,
            "max_length": args.max_length,
            "embedding_dim": args.embedding_dim,
            "hidden_channels": args.hidden_channels,
            "model_type": args.model_type,
        },
        output,
    )
    metrics = evaluate_samples(samples)
    print(f"Saved model: {output}")
    print(f"Training samples: {len(samples)}")
    print(f"Rule-span evaluation after training: F1={metrics['f1']:.4f}")
    return output


if __name__ == "__main__":
    train()
