"""
Export processed vs processed_slang experiment results into machine-readable artifacts.

This script reads:
  - results/transformer/*/metrics.json
  - results/transformer_slang/*/metrics.json

And writes:
  - reports/processed_vs_slang_results.json  (full nested payload, includes original metric dicts)
  - reports/processed_vs_slang_results.csv   (flat table for spreadsheets/LLM upload)

No third-party dependencies required.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
REPORTS_DIR = REPO_ROOT / "reports"

PROCESSED_DIR = RESULTS_DIR / "transformer"
SLANG_DIR = RESULTS_DIR / "transformer_slang"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _safe_get(d: Dict[str, Any], key: str) -> Any:
    return d.get(key)


def _flatten_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a flat dict suitable for CSV.
    Keeps the most useful scalar metrics + per-class F1, and stores confusion matrix as JSON string.
    """
    out: Dict[str, Any] = {}

    def put(name: str, value: Any) -> None:
        out[f"{prefix}{name}"] = value

    # Common fields
    put("timestamp_utc", _safe_get(metrics, "timestamp_utc"))
    put("train_samples", _safe_get(metrics, "train_samples"))
    put("val_samples", _safe_get(metrics, "val_samples"))
    put("test_samples", _safe_get(metrics, "test_samples"))
    put("max_seq_length", _safe_get(metrics, "max_seq_length"))
    put("train_batch_size", _safe_get(metrics, "train_batch_size"))
    put("eval_batch_size", _safe_get(metrics, "eval_batch_size"))
    put("num_train_epochs", _safe_get(metrics, "num_train_epochs"))
    put("learning_rate", _safe_get(metrics, "learning_rate"))
    put("weight_decay", _safe_get(metrics, "weight_decay"))
    put("warmup_ratio", _safe_get(metrics, "warmup_ratio"))
    put("seed", _safe_get(metrics, "seed"))
    put("num_parameters", _safe_get(metrics, "num_parameters"))
    put("model_size_mb", _safe_get(metrics, "model_size_mb"))

    # Transformer naming fields (if present)
    put("model_name", _safe_get(metrics, "model_name"))
    put("model_slug", _safe_get(metrics, "model_slug"))

    # Primary metrics (transformer schema)
    put("accuracy_test", _safe_float(_safe_get(metrics, "accuracy_test")))
    put("macro_f1_test", _safe_float(_safe_get(metrics, "macro_f1_test")))
    put("accuracy_val", _safe_float(_safe_get(metrics, "accuracy_val")))
    put("macro_f1_val", _safe_float(_safe_get(metrics, "macro_f1_val")))

    f1_test = _safe_get(metrics, "f1_per_class_test") or {}
    f1_val = _safe_get(metrics, "f1_per_class_val") or {}
    for label in ["T1", "T2", "T3", "T4"]:
        put(f"f1_{label}_test", _safe_float(f1_test.get(label)))
        put(f"f1_{label}_val", _safe_float(f1_val.get(label)))

    cm_test = _safe_get(metrics, "confusion_matrix_test")
    if cm_test is not None:
        put("confusion_matrix_test_json", json.dumps(cm_test, ensure_ascii=False))

    # Baseline schema (accuracy/macro_f1 without _test suffix)
    if out.get(f"{prefix}accuracy_test") is None and "accuracy" in metrics:
        put("accuracy_test", _safe_float(metrics.get("accuracy")))
    if out.get(f"{prefix}macro_f1_test") is None and "macro_f1" in metrics:
        put("macro_f1_test", _safe_float(metrics.get("macro_f1")))
    if not any(out.get(f"{prefix}f1_{lbl}_test") is not None for lbl in ["T1", "T2", "T3", "T4"]) and "f1_per_class" in metrics:
        f1 = metrics.get("f1_per_class") or {}
        for label in ["T1", "T2", "T3", "T4"]:
            put(f"f1_{label}_test", _safe_float(f1.get(label)))
    if out.get(f"{prefix}confusion_matrix_test_json") is None and "confusion_matrix" in metrics:
        put("confusion_matrix_test_json", json.dumps(metrics.get("confusion_matrix"), ensure_ascii=False))

    return out


@dataclass(frozen=True)
class ModelPair:
    model_slug: str
    processed_metrics_path: Optional[Path]
    slang_metrics_path: Optional[Path]


def _collect_pairs(processed_root: Path, slang_root: Path) -> List[ModelPair]:
    processed = {p.parent.name: p for p in processed_root.glob("*/metrics.json") if p.is_file()}
    slang = {p.parent.name: p for p in slang_root.glob("*/metrics.json") if p.is_file()}
    all_slugs = sorted(set(processed) | set(slang))
    return [
        ModelPair(
            model_slug=slug,
            processed_metrics_path=processed.get(slug),
            slang_metrics_path=slang.get(slug),
        )
        for slug in all_slugs
    ]


def _compute_delta(
    processed_metrics: Optional[Dict[str, Any]],
    slang_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Delta = slang - processed for key test metrics, if both are present.
    """
    if not processed_metrics or not slang_metrics:
        return {}

    def dnum(key: str) -> Optional[float]:
        a = _safe_float(processed_metrics.get(key))
        b = _safe_float(slang_metrics.get(key))
        if a is None or b is None:
            return None
        return b - a

    out: Dict[str, Any] = {
        "accuracy_test": dnum("accuracy_test") if "accuracy_test" in processed_metrics else dnum("accuracy"),
        "macro_f1_test": dnum("macro_f1_test") if "macro_f1_test" in processed_metrics else dnum("macro_f1"),
    }

    # Per-class deltas (transformer schema)
    p_f1 = processed_metrics.get("f1_per_class_test") or processed_metrics.get("f1_per_class") or {}
    s_f1 = slang_metrics.get("f1_per_class_test") or slang_metrics.get("f1_per_class") or {}
    per_class: Dict[str, Optional[float]] = {}
    for label in ["T1", "T2", "T3", "T4"]:
        pa = _safe_float(p_f1.get(label))
        sb = _safe_float(s_f1.get(label))
        per_class[label] = None if pa is None or sb is None else (sb - pa)
    out["f1_per_class_test"] = per_class

    return out


def main() -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = _collect_pairs(PROCESSED_DIR, SLANG_DIR)

    models_payload: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for pair in pairs:
        processed_metrics = _read_json(pair.processed_metrics_path) if pair.processed_metrics_path else None
        slang_metrics = _read_json(pair.slang_metrics_path) if pair.slang_metrics_path else None

        # Attempt to get human-readable name from available metrics
        model_name = None
        if processed_metrics and "model_name" in processed_metrics:
            model_name = processed_metrics.get("model_name")
        elif slang_metrics and "model_name" in slang_metrics:
            model_name = slang_metrics.get("model_name")

        payload = {
            "model_slug": pair.model_slug,
            "model_name": model_name,
            "processed": {
                "metrics_path": str(pair.processed_metrics_path.relative_to(REPO_ROOT)) if pair.processed_metrics_path else None,
                "metrics": processed_metrics,
            },
            "processed_slang": {
                "metrics_path": str(pair.slang_metrics_path.relative_to(REPO_ROOT)) if pair.slang_metrics_path else None,
                "metrics": slang_metrics,
            },
            "delta_slang_minus_processed": _compute_delta(processed_metrics or {}, slang_metrics or {}),
        }
        models_payload.append(payload)

        flat = {
            "model_slug": pair.model_slug,
            "model_name": model_name,
        }
        if processed_metrics:
            flat.update(_flatten_metrics("processed__", processed_metrics))
        if slang_metrics:
            flat.update(_flatten_metrics("slang__", slang_metrics))

        # Add a few common deltas (if present)
        p_mf1 = flat.get("processed__macro_f1_test")
        s_mf1 = flat.get("slang__macro_f1_test")
        flat["delta__macro_f1_test"] = None if p_mf1 is None or s_mf1 is None else (s_mf1 - p_mf1)

        p_acc = flat.get("processed__accuracy_test")
        s_acc = flat.get("slang__accuracy_test")
        flat["delta__accuracy_test"] = None if p_acc is None or s_acc is None else (s_acc - p_acc)

        for label in ["T1", "T2", "T3", "T4"]:
            pa = flat.get(f"processed__f1_{label}_test")
            sb = flat.get(f"slang__f1_{label}_test")
            flat[f"delta__f1_{label}_test"] = None if pa is None or sb is None else (sb - pa)

        csv_rows.append(flat)

    out_json = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "sources": {
            "processed_dir": str(PROCESSED_DIR.relative_to(REPO_ROOT)),
            "processed_slang_dir": str(SLANG_DIR.relative_to(REPO_ROOT)),
        },
        "models": models_payload,
    }

    json_path = REPORTS_DIR / "processed_vs_slang_results.json"
    csv_path = REPORTS_DIR / "processed_vs_slang_results.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    # Stable column order: common identifiers first, then sorted remainder
    all_keys: List[str] = []
    key_set = set()
    for r in csv_rows:
        for k in r.keys():
            if k not in key_set:
                all_keys.append(k)
                key_set.add(k)
    # Move identifiers to front
    front = ["model_slug", "model_name"]
    rest = sorted([k for k in all_keys if k not in front])
    fieldnames = front + rest

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"Models exported: {len(models_payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


