from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.font_pipeline.svg_utils import read_svg_path


COMMAND_RE = re.compile(
    r"([MmLlHhVvCcQqSsTtAaZz])|(-?\d*\.?\d+(?:e[-+]?\d+)?)"
)


def tokenize_svg_path(path_d: str):
    """
    SVG path 문자열을 command와 number token으로 분리합니다.

    예:
    M 12.0 10.5 L 14.2 13.7
    ->
    ["M", "12.0", "10.5", "L", "14.2", "13.7"]
    """
    toks = []

    for m in COMMAND_RE.finditer(path_d):
        cmd, num = m.groups()

        if cmd:
            toks.append(cmd.upper())
        elif num:
            toks.append(num)

    return toks


def float_to_int_token(v: float, scale: int = 10) -> str:
    """
    12.0 -> 120
    10.5 -> 105
    14.2 -> 142

    50x50 좌표계에서 소수점 한 자리 정밀도를 정수로 표현합니다.
    """
    return str(int(round(v * scale)))


def skeleton_path_to_integer_polyline(
    path_d: str,
    scale: int = 10,
    convert_c_to_l: bool = True,
) -> str:
    """
    skeleton SVG path를 정수 polyline sequence로 변환합니다.

    입력:
    M 12.0 10.5 L 14.2 13.7 L 18.0 20.1

    출력:
    M 120 105 L 142 137 L 180 201

    만약 C command가 들어오면:
    C x1 y1 x2 y2 x y

    convert_c_to_l=True일 때 마지막 endpoint인 x y만 사용해서:
    L x y

    로 변환합니다.
    """
    toks = tokenize_svg_path(path_d)

    out = []
    i = 0

    while i < len(toks):
        t = toks[i]

        if t == "M":
            if i + 2 >= len(toks):
                break

            x = float(toks[i + 1])
            y = float(toks[i + 2])

            out.extend([
                "M",
                float_to_int_token(x, scale),
                float_to_int_token(y, scale),
            ])

            i += 3

        elif t == "L":
            if i + 2 >= len(toks):
                break

            x = float(toks[i + 1])
            y = float(toks[i + 2])

            out.extend([
                "L",
                float_to_int_token(x, scale),
                float_to_int_token(y, scale),
            ])

            i += 3

        elif t == "C":
            if i + 6 >= len(toks):
                break

            if convert_c_to_l:
                # C x1 y1 x2 y2 x y 에서 마지막 endpoint x y만 사용
                x = float(toks[i + 5])
                y = float(toks[i + 6])

                out.extend([
                    "L",
                    float_to_int_token(x, scale),
                    float_to_int_token(y, scale),
                ])

            # C command 전체 7개 token 소비
            i += 7

        elif t == "Z":
            out.append("Z")
            i += 1

        else:
            # 예상하지 못한 숫자나 command는 건너뜀
            i += 1
    # 전체 skeleton sequence의 끝을 명시
    # --- 이 줄만 뺴면 END 만 빠짐 기존 형식 유지 
    out.append("END")

    return " ".join(out)


def build_pairs(outline_dir: Path, skeleton_dir: Path, coord_scale: int = 10):
    samples = []

    for outline_file in sorted(outline_dir.glob("*.svg")):
        tgt = skeleton_dir / outline_file.name

        if not tgt.exists():
            continue

        out_d, w, h = read_svg_path(outline_file)
        sk_d, _, _ = read_svg_path(tgt)

        sk_polyline = skeleton_path_to_integer_polyline(
            sk_d,
            scale=coord_scale,
            convert_c_to_l=True,
        )

        samples.append({
            "id": outline_file.stem,

            # source는 일단 기존 raw outline path 유지
            "src_text": f"outline2skeleton: outline path: {out_d}",

            # target만 정수화된 skeleton polyline으로 변경
            "tgt_text": f"skeleton polyline: {sk_polyline}",

            "width": w,
            "height": h,
            "coord_scale": coord_scale,
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
    ap.add_argument("--out_dir", default="dataset/processed_polyline")
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--val_size", type=float, default=0.1)

    # 10이면 12.3 -> 123
    # 100이면 12.34 -> 1234
    ap.add_argument("--coord_scale", type=int, default=10)

    args = ap.parse_args()

    rows = build_pairs(
        Path(args.outline_dir),
        Path(args.skeleton_dir),
        coord_scale=args.coord_scale,
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

    if len(train_rows) > 0:
        print("\n[DEBUG SAMPLE]")
        print("id:", train_rows[0]["id"])
        print("src_text:", train_rows[0]["src_text"][:300])
        print("tgt_text:", train_rows[0]["tgt_text"])