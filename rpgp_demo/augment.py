from typing import Dict, List


Sample = Dict[str, object]


SHORT_TEXT_TEMPLATES = [
    ("{subject}由{object}创立", "创始人", [
        ("小米公司", "雷军"),
        ("百度公司", "李彦宏"),
        ("阿里巴巴", "马云"),
    ]),
    ("{subject}在{object}创立", "创立地点", [
        ("百度公司", "北京"),
        ("小米公司", "北京"),
        ("阿里巴巴", "杭州"),
    ]),
    ("{subject}于{object}创立", "创立时间", [
        ("百度公司", "2000年"),
        ("小米公司", "2010年"),
        ("阿里巴巴", "1999年"),
    ]),
    ("{subject}由{object}提出", "提出方法", [
        ("BERT模型", "Devlin"),
        ("Transformer", "Vaswani"),
        ("RPGP方法", "本文"),
    ]),
    ("{subject}位于{object}", "位于", [
        ("复旦大学", "中国"),
        ("北京大学", "中国"),
        ("清华大学", "中国"),
    ]),
    ("{subject}出生于{object}", "出生地", [
        ("李四光", "湖北"),
        ("钱学森", "上海"),
        ("鲁迅", "浙江"),
    ]),
    ("{subject}毕业于{object}", "毕业院校", [
        ("钱学森", "交通大学"),
        ("鲁迅", "仙台医专"),
        ("李四光", "伯明翰大学"),
    ]),
]


def generate_short_text_samples(limit: int = 84) -> List[Sample]:
    samples: List[Sample] = []
    max_pairs = max(len(pairs) for _template, _relation, pairs in SHORT_TEXT_TEMPLATES)
    for pair_index in range(max_pairs):
        for template, relation, pairs in SHORT_TEXT_TEMPLATES:
            if pair_index >= len(pairs):
                continue
            subject, obj = pairs[pair_index]
            text = template.format(subject=subject, object=obj)
            if len(text) > 20:
                continue
            samples.append({
                "text": text,
                "triples": [{
                    "subject": subject,
                    "relation": relation,
                    "object": obj,
                }],
            })
            if len(samples) >= limit:
                return samples
    return samples
