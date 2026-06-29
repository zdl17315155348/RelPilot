from dataclasses import asdict, dataclass
from typing import List


@dataclass(frozen=True)
class RelationPrediction:
    name: str
    confidence: float


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int
    role: str


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    confidence: float
    subject_span: Span
    object_span: Span


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    predicted_relations: List[RelationPrediction]
    triples: List[Triple]
    spans: List[Span]

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Example:
    title: str
    text: str
