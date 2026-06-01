from __future__ import annotations

import argparse
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.font_pipeline.svg_utils import read_svg_path


def build_pairs(outline_dir: Path, skeleton_dir: Path):
    samples = []

    for outline_file in sorted(outline_dir.glob("*.svg")):
        tgt = skeleton_dir / outline_file.name

        if not tgt.exists():
            continue

        out_d, w, h = read_svg_path(outline_file)
        sk_d, _, _ = read_svg_path(tgt)

        samples.append({
            "id": outline_file.stem,
            "src_text": f"outline2skeleton: outline path: {out_d}",
            "tgt_text": f"skeleton path: {sk_d}",
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
    ap.add_argument("--out_dir", default="dataset/processed_raw")
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--val_size", type=float, default=0.1)

    args = ap.parse_args()

    rows = build_pairs(
        Path(args.outline_dir),
        Path(args.skeleton_dir)
    )

    train_rows, test_rows = train_test_split(
        rows,
        test_size=args.test_size,
        random_state=42
    )

    train_rows, val_rows = train_test_split(
        train_rows,
        test_size=args.val_size,
        random_state=42
    )

    out_dir = Path(args.out_dir)

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)

    print(
        f"Saved: train={len(train_rows)}, "
        f"val={len(val_rows)}, "
        f"test={len(test_rows)}"
    )