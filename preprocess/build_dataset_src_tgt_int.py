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
    """
    return str(int(round(v * scale)))


def outline_path_to_integer_path(
    path_d: str,
    scale: int = 10,
) -> str:
    """
    outline SVG path를 integer path로 변환합니다.

    입력:
    M 11.6 10.1 L 11.6 42.9 C 30.4 42.9 32.9 42.6 34.8 41.9 Z

    출력:
    M 116 101 L 116 429 C 304 429 329 426 348 419 Z

    source outline은 실제 곡선 정보가 중요하므로 C command를 유지합니다.
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

            x1 = float(toks[i + 1])
            y1 = float(toks[i + 2])
            x2 = float(toks[i + 3])
            y2 = float(toks[i + 4])
            x = float(toks[i + 5])
            y = float(toks[i + 6])

            out.extend([
                "C",
                float_to_int_token(x1, scale),
                float_to_int_token(y1, scale),
                float_to_int_token(x2, scale),
                float_to_int_token(y2, scale),
                float_to_int_token(x, scale),
                float_to_int_token(y, scale),
            ])

            i += 7

        elif t == "Q":
            if i + 4 >= len(toks):
                break

            x1 = float(toks[i + 1])
            y1 = float(toks[i + 2])
            x = float(toks[i + 3])
            y = float(toks[i + 4])

            out.extend([
                "Q",
                float_to_int_token(x1, scale),
                float_to_int_token(y1, scale),
                float_to_int_token(x, scale),
                float_to_int_token(y, scale),
            ])

            i += 5

        elif t == "Z":
            # Z는 close path라는 geometry command이므로 source에서는 유지
            out.append("Z")
            i += 1

        else:
            # 현재 데이터에서 H/V/A/S/T 등이 거의 없다면 일단 skip
            # 필요하면 나중에 command별 처리를 추가
            i += 1

    return " ".join(out)


def skeleton_path_to_integer_polyline(
    path_d: str,
    scale: int = 10,
    convert_c_to_l: bool = True,
    add_end: bool = True,
) -> str:
    """
    skeleton SVG path를 integer polyline sequence로 변환합니다.

    입력:
    M 12.0 10.5 L 14.2 13.7 L 18.0 20.1

    출력:
    M 120 105 L 142 137 L 180 201 END

    skeleton target은 polyline으로 만들기 위해 C가 있으면 endpoint만 사용해서 L로 변환합니다.
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

            i += 7

        elif t == "Z":
            # skeleton이 닫힌 path라면 Z 유지
            # 일반적인 중심선 skeleton이라면 Z는 거의 없어야 함
            out.append("Z")
            i += 1

        else:
            i += 1

    if add_end:
        out.append("END")

    return " ".join(out)


def build_pairs(
    outline_dir: Path,
    skeleton_dir: Path,
    coord_scale: int = 10,
):
    samples = []

    for outline_file in sorted(outline_dir.glob("*.svg")):
        tgt = skeleton_dir / outline_file.name

        if not tgt.exists():
            continue

        out_d, w, h = read_svg_path(outline_file)
        sk_d, _, _ = read_svg_path(tgt)

        outline_int = outline_path_to_integer_path(
            out_d,
            scale=coord_scale,
        )

        skeleton_int = skeleton_path_to_integer_polyline(
            sk_d,
            scale=coord_scale,
            convert_c_to_l=True,
            add_end=True,
        )

        samples.append({
            "id": outline_file.stem,

            # input도 integer path로 변경
            "src_text": f"outline2skeleton: outline path: {outline_int}",

            # output도 integer polyline + END
            "tgt_text": f"skeleton path: {skeleton_int}",

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
    ap.add_argument("--out_dir", default="dataset/processed_integer_path")
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
        print("src_text:", train_rows[0]["src_text"][:500])
        print("tgt_text:", train_rows[0]["tgt_text"])