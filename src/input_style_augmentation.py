"""
Input style augmentation for NLP multi-type project.

- Operates on processed JSONL (e.g., data/processed/train_4class.jsonl)
- T1 (Human Original): add slang + light typos
- T3 (Human Paraphrased): add light slang (no typos)
- T2/T4: unchanged

Usage (from repo root):
    python -m src.input_style_augmentation \
        --input data/processed/train_4class.jsonl \
        --output data/processed/train_4class_augmented.jsonl \
        --t1_aug_per_sample 1 \
        --t3_aug_per_sample 1
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List


# -------------------------
# Slang dictionaries
# -------------------------

SLANG_REPLACEMENTS: Dict[str, str] = {
    # common casual replacements
    "you": "u",
    "are": "r",
    "your": "ur",
    "really": "rly",
    "people": "ppl",
    "before": "b4",
    "okay": "ok",
    "great": "gr8",
    "favorite": "fav",
    "thanks": "thx",
    "sorry": "sry",
    "because": "cuz",
    "about": "abt",
    "picture": "pic",
    "tomorrow": "tmr",
}

SLANG_INSERTIONS: List[str] = [
    "lol",
    "idk",
    "tbh",
    "btw",
    "kinda",
    "sorta",
]


def _split_word_punct(token: str) -> Tuple[str, str]:
    """
    Split token into (word, trailing_punct).
    E.g., "people." -> ("people", ".")
    """
    m = re.match(r"^([A-Za-z']+)(\W*)$", token)
    if not m:
        return token, ""
    return m.group(1), m.group(2)


# -------------------------
# Slang injection
# -------------------------

def inject_slang(
    text: str,
    rng: random.Random,
    replace_prob: float,
    insert_prob: float,
    max_insertions: int,
) -> str:
    """
    Inject slang into the sentence via:
    - token-level replacements (you→u, people→ppl, ...)
    - optional insertion of standalone slang tokens (lol, tbh, ...)

    Parameters are tuned differently for T1 vs T3.
    """
    if not text:
        return text

    tokens = text.split()

    # 1) replacements
    for i, tok in enumerate(tokens):
        word, punct = _split_word_punct(tok)
        key = word.lower()
        if key in SLANG_REPLACEMENTS and rng.random() < replace_prob:
            slang = SLANG_REPLACEMENTS[key]
            tokens[i] = slang + punct

    # 2) insertions
    if max_insertions > 0 and rng.random() < insert_prob:
        num_inserted = 0
        positions = list(range(len(tokens) + 1))
        rng.shuffle(positions)
        for pos in positions:
            if num_inserted >= max_insertions:
                break
            slang = rng.choice(SLANG_INSERTIONS)
            tokens.insert(pos, slang)
            num_inserted += 1

    return " ".join(tokens)


# -------------------------
# Typo injection (T1 only)
# -------------------------

def _apply_random_typo(word: str, rng: random.Random) -> str:
    """
    Apply a single small typo to a word:
    - swap two adjacent letters
    - drop one letter
    - duplicate one letter
    """
    if len(word) < 4:
        return word

    ops = ["swap", "drop", "dup"]
    op = rng.choice(ops)
    chars = list(word)

    if op == "swap" and len(chars) >= 2:
        i = rng.randrange(len(chars) - 1)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    if op == "drop" and len(chars) >= 2:
        i = rng.randrange(len(chars))
        return "".join(chars[:i] + chars[i + 1 :])

    # default: duplicate
    i = rng.randrange(len(chars))
    return "".join(chars[: i + 1] + [chars[i]] + chars[i + 1 :])


def inject_typos(
    text: str,
    rng: random.Random,
    max_typos: int = 1,
) -> str:
    """
    Inject a very small number of typos into the sentence.
    Only modifies alphabetic tokens with length >= 4.
    """
    if not text or max_typos <= 0:
        return text

    tokens = text.split()
    candidate_indices = []
    for i, tok in enumerate(tokens):
        word, _ = _split_word_punct(tok)
        if re.fullmatch(r"[A-Za-z']{4,}", word):
            candidate_indices.append(i)

    if not candidate_indices:
        return text

    rng.shuffle(candidate_indices)
    num_applied = 0

    for idx in candidate_indices:
        if num_applied >= max_typos:
            break
        tok = tokens[idx]
        word, punct = _split_word_punct(tok)
        mutated = _apply_random_typo(word, rng)
        tokens[idx] = mutated + punct
        num_applied += 1

    return " ".join(tokens)


# -------------------------
# Row-level augmentation
# -------------------------

def update_length_fields(row: Dict[str, Any]) -> None:
    text = row.get("text", "") or ""
    row["text_len_char"] = len(text)
    row["text_len_word"] = len(text.split())


def augment_row(
    row: Dict[str, Any],
    rng: random.Random,
    t1_aug_per_sample: int,
    t3_aug_per_sample: int,
) -> List[Dict[str, Any]]:
    """
    Given a single JSONL row, return [original, augmented...]
    according to its label.

    - T1: slang + typo (augment t1_aug_per_sample times)
    - T3: slang only (augment t3_aug_per_sample times)
    - T2/T4: no augmentation
    """
    label = row.get("label")
    base_id = row.get("id", "")
    text = row.get("text", "") or ""

    # always recalc lengths for original
    update_length_fields(row)

    outputs = [row]

    if label == "T1" and t1_aug_per_sample > 0:
        for k in range(t1_aug_per_sample):
            aug = dict(row)  # shallow copy
            aug_text = inject_slang(
                text,
                rng=rng,
                replace_prob=0.4,   # stronger slang for T1
                insert_prob=0.6,
                max_insertions=2,
            )
            aug_text = inject_typos(
                aug_text,
                rng=rng,
                max_typos=1,        # at most 1 typo per sentence
            )
            aug["text"] = aug_text
            aug["id"] = f"{base_id}__aug_t1_{k+1}"
            update_length_fields(aug)
            outputs.append(aug)

    elif label == "T3" and t3_aug_per_sample > 0:
        for k in range(t3_aug_per_sample):
            aug = dict(row)
            aug_text = inject_slang(
                text,
                rng=rng,
                replace_prob=0.2,   # milder slang for T3
                insert_prob=0.3,
                max_insertions=1,
            )
            # no typos for T3
            aug["text"] = aug_text
            aug["id"] = f"{base_id}__aug_t3_{k+1}"
            update_length_fields(aug)
            outputs.append(aug)

    return outputs


# -------------------------
# Main pipeline
# -------------------------

def process_file(
    input_path: Path,
    output_path: Path,
    t1_aug_per_sample: int,
    t3_aug_per_sample: int,
    seed: int,
) -> None:
    rng = random.Random(seed)

    num_rows = 0
    num_t1_aug = 0
    num_t3_aug = 0

    with input_path.open("r", encoding="utf-8") as fin, \
            output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("label")

            augmented_rows = augment_row(
                row=row,
                rng=rng,
                t1_aug_per_sample=t1_aug_per_sample,
                t3_aug_per_sample=t3_aug_per_sample,
            )

            for r in augmented_rows:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")

            num_rows += 1
            if label == "T1":
                num_t1_aug += max(0, len(augmented_rows) - 1)
            elif label == "T3":
                num_t3_aug += max(0, len(augmented_rows) - 1)

    print(f"Processed base rows: {num_rows}")
    print(f"Extra T1 augmented rows: {num_t1_aug}")
    print(f"Extra T3 augmented rows: {num_t3_aug}")
    print(f"Total output rows: {num_rows + num_t1_aug + num_t3_aug}")


def main():
    parser = argparse.ArgumentParser(
        description="Style-based input augmentation for T1/T3 (slang + typos)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file, e.g. data/processed/train_4class.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file, e.g. data/processed/train_4class_augmented.jsonl",
    )
    parser.add_argument(
        "--t1_aug_per_sample",
        type=int,
        default=1,
        help="Number of augmented variants per T1 sample (default: 1).",
    )
    parser.add_argument(
        "--t3_aug_per_sample",
        type=int,
        default=1,
        help="Number of augmented variants per T3 sample (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Inp
