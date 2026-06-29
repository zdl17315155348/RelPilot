from pathlib import Path
from typing import List

from rpgp_demo.data import RELATION_KEYWORDS
from rpgp_demo.schema import RelationPrediction
from rpgp_demo.models.torch_data import encode_text
from rpgp_demo.models.torch_model import build_relation_classifier


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "relpilot_relation.pt"


class TorchRelationPredictor:
    def __init__(self, model_path=DEFAULT_MODEL_PATH, threshold: float = 0.5):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is not installed") from exc

        self.torch = torch
        self.model_path = Path(model_path)
        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.relations = checkpoint["relations"]
        self.vocab = checkpoint["vocab"]
        self.max_length = checkpoint["max_length"]
        self.threshold = threshold
        self.model = build_relation_classifier(
            model_type=checkpoint.get("model_type", "mean"),
            vocab_size=len(self.vocab),
            relation_count=len(self.relations),
            embedding_dim=checkpoint["embedding_dim"],
            hidden_channels=checkpoint.get("hidden_channels", 48),
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @classmethod
    def is_available(cls, model_path=DEFAULT_MODEL_PATH) -> bool:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            return False
        return Path(model_path).exists()

    def predict_relations(self, text: str) -> List[RelationPrediction]:
        ids = encode_text(text, self.vocab, self.max_length)
        tensor = self.torch.tensor([ids], dtype=self.torch.long)
        with self.torch.no_grad():
            probabilities = self.torch.sigmoid(self.model(tensor))[0].tolist()
        predictions = [
            RelationPrediction(relation, round(float(probability), 4))
            for relation, probability in zip(self.relations, probabilities)
            if probability >= self.threshold and self._has_text_evidence(text, relation)
        ]
        if predictions:
            return predictions
        best_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
        best_relation = self.relations[best_index]
        if best_relation in RELATION_KEYWORDS:
            return [RelationPrediction(best_relation, round(float(probabilities[best_index]), 4))]
        return []

    def _has_text_evidence(self, text: str, relation: str) -> bool:
        keywords = RELATION_KEYWORDS.get(relation, ())
        return any(keyword in text for keyword in keywords)
