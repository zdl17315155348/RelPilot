from typing import Dict, Iterable, List, Set


def _score(predicted: List[Set[str]], gold: List[Set[str]]) -> Dict[str, float]:
    true_positive = 0
    predicted_count = 0
    gold_count = 0
    for pred_set, gold_set in zip(predicted, gold):
        true_positive += len(pred_set & gold_set)
        predicted_count += len(pred_set)
        gold_count += len(gold_set)

    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def find_best_threshold(
    probabilities: List[Dict[str, float]],
    gold_labels: List[Set[str]],
    candidates: Iterable[float],
) -> Dict[str, float]:
    best = {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for threshold in candidates:
        predicted = [
            {relation for relation, probability in item.items() if probability >= threshold}
            for item in probabilities
        ]
        metrics = _score(predicted, gold_labels)
        if metrics["f1"] > best["f1"]:
            best = {"threshold": threshold, **metrics}
    return best
