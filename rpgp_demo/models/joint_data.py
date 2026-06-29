from typing import Dict, Iterable, List

from rpgp_demo.models.torch_data import PAD_TOKEN, UNK_TOKEN, encode_text


def build_joint_vocab(texts: Iterable[str]) -> Dict[str, int]:
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for text in texts:
        for char in text:
            if char not in vocab:
                vocab[char] = len(vocab)
    return vocab


def encode_joint_sample(sample: dict, vocab: Dict[str, int], relations: List[str], max_length: int):
    relation_index = {relation: index for index, relation in enumerate(relations)}
    relation_labels = [0.0 for _relation in relations]
    subject_labels = [
        [[0.0 for _end in range(max_length)] for _start in range(max_length)]
        for _relation in relations
    ]
    object_labels = [
        [[0.0 for _end in range(max_length)] for _start in range(max_length)]
        for _relation in relations
    ]

    text = sample.get("text", "")
    for triple in sample.get("triples", []):
        relation = triple.get("relation")
        if relation not in relation_index:
            continue
        rel_id = relation_index[relation]
        subject_span = _find_span(text, str(triple.get("subject", "")), max_length)
        object_span = _find_span(text, str(triple.get("object", "")), max_length)
        if subject_span is None or object_span is None:
            continue
        relation_labels[rel_id] = 1.0
        subject_labels[rel_id][subject_span[0]][subject_span[1]] = 1.0
        object_labels[rel_id][object_span[0]][object_span[1]] = 1.0

    return {
        "input_ids": encode_text(text, vocab, max_length),
        "relation_labels": relation_labels,
        "subject_labels": subject_labels,
        "object_labels": object_labels,
        "length": min(len(text), max_length),
    }


def _find_span(text: str, value: str, max_length: int):
    if not value:
        return None
    start = text.find(value)
    if start < 0:
        return None
    end = start + len(value) - 1
    if end >= max_length:
        return None
    return start, end
