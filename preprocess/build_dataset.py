from __future__ import annotations

import argparse
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.font_pipeline.svg_utils import read_svg_path, tokenize_path, quantize_coordinates


def build_pairs(outline_dir: Path, skeleton_dir: Path, bins: int):
    samples = []
    for outline_file in sorted(outline_dir.glob("*.svg")):
        tgt = skeleton_dir / outline_file.name
        if not tgt.exists():
            continue
        out_d, w, h = read_svg_path(outline_file)
        sk_d, _, _ = read_svg_path(tgt)
        out_tok = quantize_coordinates(tokenize_path(out_d), w, h, bins=bins)
        sk_tok = quantize_coordinates(tokenize_path(sk_d), w, h, bins=bins)
        samples.append({
            "id": outline_file.stem,
            "outline_tokens": out_tok,
            "skeleton_tokens": sk_tok,
            "width": w,
            "height": h,
        })
    return samples


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outline_dir", required=True)
    ap.add_argument("--skeleton_dir", required=True)
    ap.add_argument("--out_dir", default="dataset/processed")
    ap.add_argument("--bins", type=int, default=256)
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--val_size", type=float, default=0.1)
    args = ap.parse_args()

    rows = build_pairs(Path(args.outline_dir), Path(args.skeleton_dir), args.bins)
    train_rows, test_rows = train_test_split(rows, test_size=args.test_size, random_state=42)
    train_rows, val_rows = train_test_split(train_rows, test_size=args.val_size, random_state=42)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)
    print(f"Saved: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")
