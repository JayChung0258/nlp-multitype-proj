import json
import argparse
import unicodedata
import re
import random
from typing import Dict, Any, Iterable

# 假设 processed 数据里字段名如下，如不一致可以在这里改：
TEXT_FIELD = "text"
LABEL_FIELD = "label"
ID_FIELD = "id"

# Type1 / Type3 = human classes, per README:
# T1 -> 0, T2 -> 1, T3 -> 2, T4 -> 3
HUMAN_LABELS = {0, 2}


def normalize_text(text: str) -> str:
    """
    Basic, label-agnostic normalization.

    - Unicode NFKC
    - Lowercase
    - Collapse repeated whitespace
    - Strip leading/trailing whitespace
    """
    if not isinstance(text, str):
        return text

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def word_drop_augmentation(
    text: str,
    drop_prob: float = 0.1,
    max_drop: int = 3,
    rng: random.Random | None = None,
) -> str:
    """
    Simple word-drop augmentation:
    - Randomly remove up to `max_drop` tokens with probability `drop_prob`.
    - Only drops tokens that look like words (not pure punctuation).
    """
    if rng is None:
        rng = random

    tokens = text.split()
    if len(tokens) <= 5:
        # For very short sentences, avoid aggressive drops
        return text

    # Candidate indices to drop: non-punctuation, length > 3
    candidate_indices = [
        i for i, tok in enumerate(tokens)
        if re.search(r"\w", tok) and len(tok) > 3
    ]
    rng.shuffle(candidate_indices)

    num_to_drop = 0
    for idx in candidate_indices:
        if num_to_drop >= max_drop:
            break
        if rng.random() < drop_prob:
            tokens[idx] = None
            num_to_drop += 1

    kept_tokens = [t for t in tokens if t is not None]
    if not kept_tokens:
        # Fallback: if we dropped everything by accident
        return text

    return " ".join(kept_tokens)


def process_jsonl(
    input_path: str,
    output_path: str,
    augment_factor: int = 1,
    seed: int = 42,
) -> None:
    """
    Read a processed 4-class JSONL file, normalize text for all samples,
    and augment human classes (T1/T3) with word-drop variants.

    Parameters
    ----------
    input_path : str
        Path to original JSONL (e.g., data/processed/train_4class.jsonl).
    output_path : str
        Path to write modified JSONL.
    augment_factor : int
        For each human sample, how many augmented variants to generate.
        0 = no augmentation, just normalization.
    seed : int
        Random seed for reproducibility.
    """
    rng = random.Random(seed)

    num_original = 0
    num_augmented = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            record: Dict[str, Any] = json.loads(line)
            label = record.get(LABEL_FIELD)

            # 1) Normalize text for everyone
            text = record.get(TEXT_FIELD, "")
            norm_text = normalize_text(text)
            record[TEXT_FIELD] = norm_text

            # Write the normalized original
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_original += 1

            # 2) Augment only human classes (Type1 / Type3)
            if augment_factor > 0 and label in HUMAN_LABELS:
                for k in range(augment_factor):
                    aug_record = dict(record)  # shallow copy is enough

                    aug_text = word_drop_augmentation(
                        norm_text,
                        drop_prob=0.1,
                        max_drop=3,
                        rng=rng,
                    )
                    aug_record[TEXT_FIELD] = aug_text

                    # If there's an ID, make it unique
                    if ID_FIELD in aug_record and isinstance(aug_record[ID_FIELD], str):
                        aug_record[ID_FIELD] = f"{aug_record[ID_FIELD]}_aug{k+1}"

                    fout.write(json.dumps(aug_record, ensure_ascii=False) + "\n")
                    num_augmented += 1

    print(f"Done. Wrote {num_original} normalized records "
          f"and {num_augmented} augmented records to {output_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Input modification for 4-class text classification "
                    "(normalize + human-class augmentation)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file (e.g., data/processed/train_4class.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file (e.g., data/processed/train_4class_human_augmented.jsonl)",
    )
    parser.add_argument(
        "--augment_factor",
        type=int,
        default=1,
        help="How many augmented variants per human sample (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()
    process_jsonl(
        input_path=args.input,
        output_path=args.output,
        augment_factor=args.augment_factor,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
