"""
Input style overwrite for NLP Multi-Type project (NO data size increase).

- Works on processed JSONL files produced by src.data_prep
  (e.g., data/processed/train_4class.jsonl)
- Follows the DATA_CONTRACT schema, including:
  id, family_id, source, text, label, label_id, text_len_char, text_len_word, ...

Overwrite rules (NO extra rows):
- T1 (Human Original):
    - slang replacements + optional insertions
    - at most 1 small character-level typo per sentence
- T3 (Human Paraphrased):
    - light slang only
    - no typos
- T2/T4:
    - unchanged

Usage (from repo root):
    python -m src.input_style_overwrite \
        --input data/processed/train_4class.jsonl \
        --output data/processed/train_4class_style.jsonl \
        --seed 42

Repeat for val/test with the same seed.

Notes:
- Output JSONL row count == input JSONL row count.
- No class distribution skew (no oversampling).
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
        return "".join(chars[:i] + chars[i + 1:])

    # default: duplicate
    i = rng.randrange(len(chars))
    return "".join(chars[:i + 1] + [chars[i]] + chars[i + 1:])


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
# Row-level overwrite
# -------------------------

def update_length_fields(row: Dict[str, Any]) -> None:
    """
    Keep DATA_CONTRACT fields consistent:
    - text_len_char
    - text_len_word
    """
    text = row.get("text", "") or ""
    row["text_len_char"] = len(text)
    row["text_len_word"] = len(text.split())


def overwrite_row_style(
    row: Dict[str, Any],
    rng: random.Random,
    # T1 params
    t1_replace_prob: float,
    t1_insert_prob: float,
    t1_max_insertions: int,
    t1_max_typos: int,
    # T3 params
    t3_replace_prob: float,
    t3_insert_prob: float,
    t3_max_insertions: int,
    # optional metadata
    add_meta: bool = True,
) -> Dict[str, Any]:
    """
    Overwrite row['text'] for T1/T3 only.
    Output row count stays EXACTLY the same as input.
    """
    label_str = row.get("label")
    label_id = row.get("label_id")
    text = row.get("text", "") or ""

    # robust check: support both string labels and numeric ids
    is_t1 = (label_str == "T1") or (label_id == 0)
    is_t3 = (label_str == "T3") or (label_id == 2)

    aug_tag = "none"

    if is_t1:
        text = inject_slang(
            text,
            rng=rng,
            replace_prob=t1_replace_prob,
            insert_prob=t1_insert_prob,
            max_insertions=t1_max_insertions,
        )
        text = inject_typos(
            text,
            rng=rng,
            max_typos=t1_max_typos,
        )
        row["text"] = text
        aug_tag = "t1_slang+typo"

    elif is_t3:
        text = inject_slang(
            text,
            rng=rng,
            replace_prob=t3_replace_prob,
            insert_prob=t3_insert_prob,
            max_insertions=t3_max_insertions,
        )
        row["text"] = text
        aug_tag = "t3_light_slang"

    update_length_fields(row)

    if add_meta:
        # helpful for analysis; doesn't affect label distribution
        row["style_aug"] = aug_tag

    return row


# -------------------------
# Main pipeline
# -------------------------

def process_file(
    input_path: Path,
    output_path: Path,
    seed: int,
    # T1 params
    t1_replace_prob: float,
    t1_insert_prob: float,
    t1_max_insertions: int,
    t1_max_typos: int,
    # T3 params
    t3_replace_prob: float,
    t3_insert_prob: float,
    t3_max_insertions: int,
    # meta
    add_meta: bool,
) -> None:
    rng = random.Random(seed)

    num_rows = 0
    num_t1_changed = 0
    num_t3_changed = 0

    with input_path.open("r", encoding="utf-8") as fin, \
            output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            row: Dict[str, Any] = json.loads(line)

            label_str = row.get("label")
            label_id = row.get("label_id")
            is_t1 = (label_str == "T1") or (label_id == 0)
            is_t3 = (label_str == "T3") or (label_id == 2)

            row = overwrite_row_style(
                row=row,
                rng=rng,
                t1_replace_prob=t1_replace_prob,
                t1_insert_prob=t1_insert_prob,
                t1_max_insertions=t1_max_insertions,
                t1_max_typos=t1_max_typos,
                t3_replace_prob=t3_replace_prob,
                t3_insert_prob=t3_insert_prob,
                t3_max_insertions=t3_max_insertions,
                add_meta=add_meta,
            )

            if is_t1:
                num_t1_changed += 1
            elif is_t3:
                num_t3_changed += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            num_rows += 1

    print(f"Processed rows (output == input): {num_rows}")
    print(f"T1 rows overwritten:             {num_t1_changed}")
    print(f"T3 rows overwritten:             {num_t3_changed}")
    print(f"T2/T4 rows unchanged:            {num_rows - num_t1_changed - num_t3_changed}")


def main():
    parser = argparse.ArgumentParser(
        description="Style-based overwrite for T1/T3 (slang + typos) without increasing dataset size."
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
        help="Output JSONL file, e.g. data/processed/train_4class_style.jsonl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42). Use same seed across train/val/test for reproducibility.",
    )

    # T1 params
    parser.add_argument("--t1_replace_prob", type=float, default=0.4)
    parser.add_argument("--t1_insert_prob", type=float, default=0.6)
    parser.add_argument("--t1_max_insertions", type=int, default=2)
    parser.add_argument("--t1_max_typos", type=int, default=1)

    # T3 params
    parser.add_argument("--t3_replace_prob", type=float, default=0.2)
    parser.add_argument("--t3_insert_prob", type=float, default=0.3)
    parser.add_argument("--t3_max_insertions", type=int, default=1)

    # meta
    parser.add_argument(
        "--add_meta",
        action="store_true",
        help="If set, add row['style_aug'] = {none|t1_slang+typo|t3_light_slang} for analysis.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("==============================================")
    print("Input style overwrite (NO extra rows)")
    print("==============================================")
    print(f"Input:             {input_path}")
    print(f"Output:            {output_path}")
    print(f"Seed:              {args.seed}")
    print(f"T1 replace_prob:   {args.t1_replace_prob}")
    print(f"T1 insert_prob:    {args.t1_insert_prob}")
    print(f"T1 max_insertions: {args.t1_max_insertions}")
    print(f"T1 max_typos:      {args.t1_max_typos}")
    print(f"T3 replace_prob:   {args.t3_replace_prob}")
    print(f"T3 insert_prob:    {args.t3_insert_prob}")
    print(f"T3 max_insertions: {args.t3_max_insertions}")
    print(f"Add meta:          {args.add_meta}")
    print()

    process_file(
        input_path=input_path,
        output_path=output_path,
        seed=args.seed,
        t1_replace_prob=args.t1_replace_prob,
        t1_insert_prob=args.t1_insert_prob,
        t1_max_insertions=args.t1_max_insertions,
        t1_max_typos=args.t1_max_typos,
        t3_replace_prob=args.t3_replace_prob,
        t3_insert_prob=args.t3_insert_prob,
        t3_max_insertions=args.t3_max_insertions,
        add_meta=args.add_meta,
    )


if __name__ == "__main__":
    main()
