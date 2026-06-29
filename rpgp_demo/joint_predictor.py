from pathlib import Path
from typing import List

from rpgp_demo.schema import ExtractionResult, RelationPrediction, Span, Triple
from rpgp_demo.torch_data import encode_text
from rpgp_demo.joint_model import JointExtractionModel


DEFAULT_JOINT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "relpilot_joint.pt"


class JointExtractionPredictor:
    def __init__(self, model_path=DEFAULT_JOINT_MODEL_PATH):
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
        self.relation_threshold = checkpoint.get("relation_threshold", 0.5)
        self.span_threshold = checkpoint.get("span_threshold", 0.5)
        self.model = JointExtractionModel(
            vocab_size=len(self.vocab),
            relation_count=len(self.relations),
            embedding_dim=checkpoint["embedding_dim"],
            hidden_size=checkpoint["hidden_size"],
            max_length=self.max_length,
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @classmethod
    def is_available(cls, model_path=DEFAULT_JOINT_MODEL_PATH) -> bool:
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            return False
        return Path(model_path).exists()

    def extract(self, text: str) -> ExtractionResult:
        clean_text = text.strip()
        ids = encode_text(clean_text, self.vocab, self.max_length)
        tensor = self.torch.tensor([ids], dtype=self.torch.long)
        with self.torch.no_grad():
            outputs = self.model(tensor)
            relation_probs = self.torch.sigmoid(outputs["relation_logits"])[0]
            subject_probs = self.torch.sigmoid(outputs["subject_logits"])[0]
            object_probs = self.torch.sigmoid(outputs["object_logits"])[0]

        predictions = self._decode_relations(relation_probs)
        triples = self._decode_triples(clean_text, predictions, subject_probs, object_probs)
        spans = self._dedupe_spans(
            [triple.subject_span for triple in triples]
            + [triple.object_span for triple in triples]
        )
        return ExtractionResult(clean_text, predictions, triples, spans)

    def _decode_relations(self, relation_probs) -> List[RelationPrediction]:
        predictions = [
            RelationPrediction(relation, round(float(probability), 4))
            for relation, probability in zip(self.relations, relation_probs.tolist())
            if probability >= self.relation_threshold
        ]
        if predictions:
            return predictions
        best_index = int(self.torch.argmax(relation_probs).item())
        return [RelationPrediction(self.relations[best_index], round(float(relation_probs[best_index]), 4))]

    def _decode_triples(self, text: str, predictions: List[RelationPrediction], subject_probs, object_probs):
        triples = []
        text_length = min(len(text), self.max_length)
        for prediction in predictions:
            rel_id = self.relations.index(prediction.name)
            subjects = self._top_spans(text, subject_probs[rel_id], text_length, "subject")
            objects = self._top_spans(text, object_probs[rel_id], text_length, "object")
            for subject in subjects[:2]:
                for obj in objects[:2]:
                    triples.append(Triple(subject.text, prediction.name, obj.text, prediction.confidence, subject, obj))
        return self._dedupe_triples(triples)

    def _top_spans(self, text: str, matrix, text_length: int, role: str) -> List[Span]:
        spans = []
        for start in range(text_length):
            for end in range(start, text_length):
                probability = float(matrix[start][end])
                if probability >= self.span_threshold:
                    spans.append((probability, Span(text[start:end + 1], start, end + 1, role)))
        spans.sort(key=lambda item: (-item[0], item[1].start, item[1].end))
        return [span for _probability, span in spans if span.text]

    def _dedupe_triples(self, triples: List[Triple]) -> List[Triple]:
        seen = set()
        result = []
        for triple in triples:
            key = (triple.subject, triple.relation, triple.object)
            if key not in seen:
                seen.add(key)
                result.append(triple)
        return result

    def _dedupe_spans(self, spans: List[Span]) -> List[Span]:
        seen = set()
        result = []
        for span in sorted(spans, key=lambda item: (item.start, item.end, item.role)):
            key = (span.start, span.end, span.role)
            if key not in seen:
                seen.add(key)
                result.append(span)
        return result
