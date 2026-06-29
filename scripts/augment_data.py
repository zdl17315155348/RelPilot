import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rpgp_demo.augment import generate_short_text_samples


DEFAULT_INPUT = PROJECT_ROOT / "data" / "eval_samples.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "train_augmented.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build augmented short-text training data.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Base sample JSON path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path")
    parser.add_argument("--limit", type=int, default=84, help="Generated short-text sample limit")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.input, "r", encoding="utf-8") as file:
        samples = json.load(file)
    samples.extend(generate_short_text_samples(limit=args.limit))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(samples, file, ensure_ascii=False, indent=2)

    print(f"Saved augmented data: {output}")
    print(f"Samples: {len(samples)}")
    return output


if __name__ == "__main__":
    main()
