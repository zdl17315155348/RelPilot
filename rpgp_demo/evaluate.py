from typing import Dict, Iterable, Set, Tuple

from rpgp_demo.extractor import RPGPExtractor


TripleKey = Tuple[str, str, str]


def evaluate_samples(samples: Iterable[dict]) -> Dict[str, float]:
    extractor = RPGPExtractor()
    sample_count = 0
    gold_total = 0
    predicted_total = 0
    correct_total = 0
    sample_details = []

    for sample in samples:
        sample_count += 1
        gold = _triple_set(sample.get("triples", []))
        predicted = {
            (triple.subject, triple.relation, triple.object)
            for triple in extractor.extract(sample.get("text", "")).triples
        }
        gold_total += len(gold)
        predicted_total += len(predicted)
        correct = gold & predicted
        correct_total += len(correct)
        sample_details.append(
            {
                "text": sample.get("text", ""),
                "gold": _sorted_triples(gold),
                "predicted": _sorted_triples(predicted),
                "correct": _sorted_triples(correct),
                "missing": _sorted_triples(gold - predicted),
                "extra": _sorted_triples(predicted - gold),
            }
        )

    precision = correct_total / predicted_total if predicted_total else 0.0
    recall = correct_total / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sample_count": sample_count,
        "gold_count": gold_total,
        "predicted_count": predicted_total,
        "correct_count": correct_total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "samples": sample_details,
    }


def _triple_set(items: Iterable[dict]) -> Set[TripleKey]:
    return {
        (str(item["subject"]), str(item["relation"]), str(item["object"]))
        for item in items
    }


def _sorted_triples(items: Iterable[TripleKey]):
    return [list(item) for item in sorted(items)]
