import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rpgp_demo.data import RELATION_KEYWORDS
from rpgp_demo.evaluate import evaluate_samples
from rpgp_demo.joint_data import build_joint_vocab, encode_joint_sample
from rpgp_demo.joint_model import JointExtractionModel


DEFAULT_DATASET = PROJECT_ROOT / "data" / "train_augmented.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "relpilot_joint.pt"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the RelPilot joint BiLSTM-GlobalPointer model.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Training dataset JSON path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Model checkpoint path")
    parser.add_argument("--best-output", default=None, help="Best checkpoint path")
    parser.add_argument("--resume", default=None, help="Resume checkpoint path")
    parser.add_argument("--limit", type=int, default=None, help="Use the first N samples")
    parser.add_argument("--epochs", type=int, default=220, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-length", type=int, default=64, help="Max character length")
    parser.add_argument("--embedding-dim", type=int, default=96, help="Embedding dimension")
    parser.add_argument("--hidden-size", type=int, default=64, help="BiLSTM hidden size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--relation-threshold", type=float, default=0.45, help="Relation probability threshold")
    parser.add_argument("--span-threshold", type=float, default=0.60, help="Span probability threshold")
    parser.add_argument("--span-pos-weight", type=float, default=8.0, help="Positive span loss weight")
    parser.add_argument(
        "--relations",
        choices=("auto", "preset"),
        default="auto",
        help="Use dataset relations or the built-in demo relation set",
    )
    return parser.parse_args(argv)


def train(argv=None):
    args = parse_args(argv)
    torch.manual_seed(11)
    with open(args.dataset, "r", encoding="utf-8") as file:
        samples = json.load(file)
    if args.limit is not None:
        samples = samples[:args.limit]

    relations = list(RELATION_KEYWORDS) if args.relations == "preset" else _collect_relations(samples)
    vocab = build_joint_vocab(sample["text"] for sample in samples)
    dataset = JointSampleDataset(samples, vocab, relations, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_joint_batch)
    model = JointExtractionModel(
        len(vocab),
        len(relations),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        max_length=args.max_length,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    relation_criterion = nn.BCEWithLogitsLoss()
    span_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.span_pos_weight))
    start_epoch = 0
    best_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        best_loss = checkpoint.get("best_loss", best_loss)

    output = Path(args.output)
    best_output = Path(args.best_output) if args.best_output else output.with_suffix(".best.pt")

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        model.train()
        total_loss = 0.0
        for input_ids, relation_labels, subject_labels, object_labels in loader:
            outputs = model(input_ids)
            loss = (
                relation_criterion(outputs["relation_logits"], relation_labels)
                + span_criterion(outputs["subject_logits"], subject_labels)
                + span_criterion(outputs["object_logits"], object_labels)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        average_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch}: loss={average_loss:.4f}")
        checkpoint = _build_checkpoint(
            model,
            optimizer,
            relations,
            vocab,
            args,
            epoch,
            average_loss,
            best_loss,
            len(samples),
        )
        _save_checkpoint(checkpoint, output)
        if average_loss < best_loss:
            best_loss = average_loss
            checkpoint["best_loss"] = best_loss
            _save_checkpoint(checkpoint, best_output)

    metrics = evaluate_samples(samples)
    print(f"Saved joint model: {output}")
    print(f"Saved best model: {best_output}")
    print(f"Training samples: {len(samples)}")
    print(f"Demo evaluation after training: F1={metrics['f1']:.4f}")
    return output


def _collect_relations(samples):
    relations = []
    seen = set()
    for sample in samples:
        for triple in sample.get("triples", []):
            relation = triple.get("relation")
            if relation and relation not in seen:
                seen.add(relation)
                relations.append(relation)
    return relations


class JointSampleDataset(Dataset):
    def __init__(self, samples, vocab, relations, max_length):
        self.samples = samples
        self.vocab = vocab
        self.relations = relations
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return encode_joint_sample(self.samples[index], self.vocab, self.relations, self.max_length)


def collate_joint_batch(items):
    return (
        torch.tensor([item["input_ids"] for item in items], dtype=torch.long),
        torch.tensor([item["relation_labels"] for item in items], dtype=torch.float32),
        torch.tensor([item["subject_labels"] for item in items], dtype=torch.float32),
        torch.tensor([item["object_labels"] for item in items], dtype=torch.float32),
    )


def _build_checkpoint(model, optimizer, relations, vocab, args, epoch, loss, best_loss, trained_samples):
    return {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "relations": relations,
        "vocab": vocab,
        "max_length": args.max_length,
        "embedding_dim": args.embedding_dim,
        "hidden_size": args.hidden_size,
        "relation_threshold": args.relation_threshold,
        "span_threshold": args.span_threshold,
        "model_type": "bilstm_globalpointer",
        "epoch": epoch,
        "loss": loss,
        "best_loss": best_loss,
        "trained_samples": trained_samples,
    }


def _save_checkpoint(checkpoint, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)


if __name__ == "__main__":
    train()
