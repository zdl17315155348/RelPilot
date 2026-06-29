from typing import Dict, Iterable, List


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_relation_labels(sample: dict, relations: List[str]) -> Dict[str, float]:
    labels = {relation: 0.0 for relation in relations}
    for triple in sample.get("triples", []):
        relation = triple.get("relation")
        if relation in labels:
            labels[relation] = 1.0
    return labels


def build_vocab(texts: Iterable[str]) -> Dict[str, int]:
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for text in texts:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_length: int) -> List[int]:
    ids = [vocab.get(char, vocab[UNK_TOKEN]) for char in text[:max_length]]
    ids.extend([vocab[PAD_TOKEN]] * (max_length - len(ids)))
    return ids
