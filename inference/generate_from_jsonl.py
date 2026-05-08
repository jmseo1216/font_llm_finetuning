from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.font_pipeline.svg_utils import dequantize_sequence, tokens_to_svg_path


def row_to_outline_svg(row):
    d = tokens_to_svg_path(dequantize_sequence(row["outline_tokens"], row["width"], row["height"]))
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {row["width"]} {row["height"]}"><path d="{d}" fill="none" stroke="black"/></svg>'


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--sample_id", required=True)
    ap.add_argument("--output_svg", required=True)
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.jsonl).read_text(encoding="utf-8").splitlines()]
    row = next((r for r in rows if str(r["id"]) == str(args.sample_id)), None)
    if row is None:
        raise ValueError(f"sample_id={args.sample_id} not found")

    Path(args.output_svg).write_text(row_to_outline_svg(row), encoding="utf-8")
    print(args.output_svg)
