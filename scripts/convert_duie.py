import argparse
import json
from pathlib import Path


def convert_duie_jsonl(input_path, output_path, limit=None):
    samples = []
    with open(input_path, "r", encoding="utf-8") as file:
        for line in file:
            if limit is not None and len(samples) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            triples = []
            for spo in item.get("spo_list", []):
                obj = _object_value(spo.get("object", ""))
                triples.append({
                    "subject": str(spo.get("subject", "")),
                    "relation": str(spo.get("predicate", "")),
                    "object": obj,
                })
            if triples:
                samples.append({"text": item.get("text", ""), "triples": triples})

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(samples, file, ensure_ascii=False, indent=2)
    return len(samples)


def _object_value(value):
    if isinstance(value, dict):
        if "@value" in value:
            return str(value["@value"])
        for item in value.values():
            if isinstance(item, list):
                return str(item[0]) if item else ""
            return str(item)
        return ""
    return str(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Convert DuIE JSONL data to RelPilot samples.")
    parser.add_argument("--input", required=True, help="DuIE json/jsonl path")
    parser.add_argument("--output", default="data/duie_samples.json", help="Output project JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Max converted samples")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    count = convert_duie_jsonl(args.input, args.output, args.limit)
    print(f"Saved converted samples: {args.output}")
    print(f"Samples: {count}")
    return count


if __name__ == "__main__":
    main()
