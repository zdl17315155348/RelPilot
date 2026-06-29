import re
from typing import Iterable, List

from rpgp_demo.data import RELATION_KEYWORDS
from rpgp_demo.schema import ExtractionResult, RelationPrediction, Span, Triple


class RPGPExtractor:
    """Lightweight prototype of relation pre-judgement plus span extraction."""

    def __init__(self, relation_predictor=None, joint_model_path=None):
        self.joint_predictor = self._build_joint_predictor(joint_model_path)
        self.relation_predictor = relation_predictor or self._build_default_predictor()

    def extract(self, text: str) -> ExtractionResult:
        clean_text = text.strip()
        if self.joint_predictor is not None:
            result = self.joint_predictor.extract(clean_text)
            if result.triples:
                return result
        predictions = self.predict_relations(clean_text)
        triples = self.extract_triples(clean_text, [item.name for item in predictions])
        spans = self._dedupe_spans(
            [triple.subject_span for triple in triples]
            + [triple.object_span for triple in triples]
        )
        return ExtractionResult(clean_text, predictions, triples, spans)

    def predict_relations(self, text: str) -> List[RelationPrediction]:
        if self.relation_predictor is not None:
            return self.relation_predictor.predict_relations(text)
        predictions = []
        for relation, keywords in RELATION_KEYWORDS.items():
            hits = sum(1 for keyword in keywords if keyword in text)
            if hits:
                confidence = round(0.55 + 0.18 * hits, 2)
                predictions.append(RelationPrediction(relation, min(confidence, 0.96)))
        return predictions

    def _build_joint_predictor(self, model_path=None):
        try:
            from rpgp_demo.joint_predictor import JointExtractionPredictor
        except Exception:
            return None
        if model_path and not JointExtractionPredictor.is_available(model_path):
            return None
        if not model_path and not JointExtractionPredictor.is_available():
            return None
        try:
            return JointExtractionPredictor(model_path) if model_path else JointExtractionPredictor()
        except Exception:
            return None

    def _build_default_predictor(self):
        try:
            from rpgp_demo.torch_predictor import TorchRelationPredictor
        except Exception:
            return None
        if not TorchRelationPredictor.is_available():
            return None
        try:
            return TorchRelationPredictor()
        except Exception:
            return None

    def extract_triples(self, text: str, relations: Iterable[str]) -> List[Triple]:
        relation_set = set(relations)
        triples: List[Triple] = []

        if {"创始人", "创立时间", "创立地点"} & relation_set:
            triples.extend(self._extract_company_founding(text, relation_set))
        if "位于" in relation_set:
            triples.extend(self._extract_university_location(text))
        if {"改进算法", "性能评估"} & relation_set:
            triples.extend(self._extract_science_relations(text, relation_set))
        if "提出方法" in relation_set:
            triples.extend(self._extract_method_proposal(text))
        if {"出生地", "毕业院校"} & relation_set:
            triples.extend(self._extract_person_relations(text, relation_set))

        return triples

    def _extract_company_founding(self, text: str, relations: set) -> List[Triple]:
        pattern = re.search(
            r"(?P<company>[\u4e00-\u9fa5A-Za-z0-9]+(?:公司|集团|企业|机构))由"
            r"(?P<founder>[\u4e00-\u9fa5A-Za-z]+)于(?P<year>\d{4}年)在"
            r"(?P<place>[\u4e00-\u9fa5]+)创立",
            text,
        )
        if not pattern:
            return self._extract_short_company_founding(text, relations)

        company = self._span_from_match(pattern, "company", "subject")
        triples = []
        if "创始人" in relations:
            triples.append(
                self._triple(company, "创始人", self._span_from_match(pattern, "founder", "object"), 0.94)
            )
        if "创立时间" in relations:
            triples.append(
                self._triple(company, "创立时间", self._span_from_match(pattern, "year", "object"), 0.91)
            )
        if "创立地点" in relations:
            triples.append(
                self._triple(company, "创立地点", self._span_from_match(pattern, "place", "object"), 0.88)
            )
        return triples

    def _extract_short_company_founding(self, text: str, relations: set) -> List[Triple]:
        triples = []
        founder_pattern = re.search(
            r"(?P<company>[\u4e00-\u9fa5A-Za-z0-9]+(?:公司|集团|企业|机构))由"
            r"(?P<founder>[\u4e00-\u9fa5A-Za-z]+)创立",
            text,
        )
        if founder_pattern and "创始人" in relations:
            triples.append(
                self._triple(
                    self._span_from_match(founder_pattern, "company", "subject"),
                    "创始人",
                    self._span_from_match(founder_pattern, "founder", "object"),
                    0.86,
                )
            )
        place_pattern = re.search(
            r"(?P<company>[\u4e00-\u9fa5A-Za-z0-9]+(?:公司|集团|企业|机构))在"
            r"(?P<place>[\u4e00-\u9fa5]+)创立",
            text,
        )
        if place_pattern and "创立地点" in relations:
            triples.append(
                self._triple(
                    self._span_from_match(place_pattern, "company", "subject"),
                    "创立地点",
                    self._span_from_match(place_pattern, "place", "object"),
                    0.85,
                )
            )
        return triples

    def _extract_university_location(self, text: str) -> List[Triple]:
        if "中国" not in text:
            return []
        object_span = self._literal_span(text, "中国", "object")
        triples = []
        for match in re.finditer(r"[\u4e00-\u9fa5]{2,12}?大学", text):
            name = match.group().split("和")[-1].split("、")[-1]
            start = match.end() - len(name)
            subject_span = Span(name, start, match.end(), "subject")
            triples.append(self._triple(subject_span, "位于", object_span, 0.87))
        if triples:
            return triples
        direct_pattern = re.search(r"(?P<school>[\u4e00-\u9fa5]+大学)位于(?P<country>中国)", text)
        if direct_pattern:
            return [
                self._triple(
                    self._span_from_match(direct_pattern, "school", "subject"),
                    "位于",
                    self._span_from_match(direct_pattern, "country", "object"),
                    0.88,
                )
            ]
        for match in re.finditer(r"[\u4e00-\u9fa5]+大学", text):
            subject_span = Span(match.group(), match.start(), match.end(), "subject")
            triples.append(self._triple(subject_span, "位于", object_span, 0.87))
        return triples

    def _extract_science_relations(self, text: str, relations: set) -> List[Triple]:
        triples = []
        method_match = re.search(r"(?P<method>[A-Za-z0-9\u4e00-\u9fa5]+方法)", text)
        method_span = self._span_from_match(method_match, "method", "subject") if method_match else None
        if method_span and "改进算法" in relations:
            algo_match = re.search(r"(?P<algo>[A-Za-z0-9]+算法)", text)
            if algo_match:
                triples.append(
                    self._triple(method_span, "改进算法", self._span_from_match(algo_match, "algo", "object"), 0.9)
                )
        if method_span and "性能评估" in relations:
            dataset_match = re.search(r"(?P<dataset>[A-Za-z0-9]+数据集)", text)
            if dataset_match:
                triples.append(
                    self._triple(
                        method_span,
                        "性能评估",
                        self._span_from_match(dataset_match, "dataset", "object"),
                        0.86,
                    )
                )
        return triples

    def _extract_method_proposal(self, text: str) -> List[Triple]:
        pattern = re.search(
            r"(?P<method>[A-Za-z0-9\u4e00-\u9fa5]+(?:模型|方法))由(?P<author>[A-Za-z\u4e00-\u9fa5]+(?:等?人)?).*提出",
            text,
        )
        if not pattern:
            return []
        return [
            self._triple(
                self._span_from_match(pattern, "method", "subject"),
                "提出方法",
                self._span_from_match(pattern, "author", "object"),
                0.89,
            )
        ]

    def _extract_person_relations(self, text: str, relations: set) -> List[Triple]:
        person_match = re.search(r"(?P<person>[\u4e00-\u9fa5]{2,4})出生于", text)
        if not person_match:
            return self._extract_short_school_relation(text, relations)
        person_span = self._span_from_match(person_match, "person", "subject")
        triples = []
        if "出生地" in relations:
            birth_match = re.search(r"出生于(?P<place>[\u4e00-\u9fa5]+)", text)
            if birth_match:
                triples.append(
                    self._triple(person_span, "出生地", self._span_from_match(birth_match, "place", "object"), 0.88)
                )
        if "毕业院校" in relations:
            school_match = re.search(r"毕业于(?P<school>[\u4e00-\u9fa5]+大学)", text)
            if school_match:
                triples.append(
                    self._triple(
                        person_span,
                        "毕业院校",
                        self._span_from_match(school_match, "school", "object"),
                        0.86,
                    )
                )
        return triples

    def _extract_short_school_relation(self, text: str, relations: set) -> List[Triple]:
        if "毕业院校" not in relations:
            return []
        school_match = re.search(r"(?P<person>[\u4e00-\u9fa5]{2,4})毕业于(?P<school>[\u4e00-\u9fa5]+大学)", text)
        if not school_match:
            return []
        return [
            self._triple(
                self._span_from_match(school_match, "person", "subject"),
                "毕业院校",
                self._span_from_match(school_match, "school", "object"),
                0.86,
            )
        ]

    def _triple(self, subject: Span, relation: str, obj: Span, confidence: float) -> Triple:
        return Triple(subject.text, relation, obj.text, confidence, subject, obj)

    def _span_from_match(self, match: re.Match, group: str, role: str) -> Span:
        return Span(match.group(group), match.start(group), match.end(group), role)

    def _literal_span(self, text: str, value: str, role: str) -> Span:
        start = text.index(value)
        return Span(value, start, start + len(value), role)

    def _dedupe_spans(self, spans: List[Span]) -> List[Span]:
        seen = set()
        result = []
        for span in sorted(spans, key=lambda item: (item.start, item.end, item.role)):
            key = (span.start, span.end, span.role)
            if key not in seen:
                seen.add(key)
                result.append(span)
        return result
