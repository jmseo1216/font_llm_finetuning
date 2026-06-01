from __future__ import annotations

import argparse
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Optional

from sklearn.model_selection import train_test_split


COMMAND_RE = re.compile(
    r"([MmLlHhVvCcQqSsTtAaZz])|(-?\d*\.?\d+(?:e[-+]?\d+)?)"
)


def normalize_tar_name(name: str) -> str:
    """
    tar 내부 경로를 통일합니다.
    Windows 경로 구분자도 대비합니다.
    """
    return name.replace("\\", "/").lstrip("./")


def is_svg_file(name: str) -> bool:
    return normalize_tar_name(name).lower().endswith(".svg")


def get_basename(name: str) -> str:
    return Path(normalize_tar_name(name)).name


def get_stem(name: str) -> str:
    return Path(normalize_tar_name(name)).stem


def detect_group(name: str) -> Optional[str]:
    """
    tar 내부 path에서 aug_ttf / aug_fnt를 감지합니다.

    예:
    all_dataset/aug_ttf/arial_aug_0000_33.svg -> aug_ttf
    aug_fnt/simplex_aug_0000_33.svg -> aug_fnt
    """
    parts = normalize_tar_name(name).split("/")

    if "aug_ttf" in parts:
        return "aug_ttf"

    if "aug_fnt" in parts:
        return "aug_fnt"

    return None


def read_tar_member_text(tf: tarfile.TarFile, member: tarfile.TarInfo) -> str:
    f = tf.extractfile(member)

    if f is None:
        raise RuntimeError(f"Failed to read tar member: {member.name}")

    data = f.read()

    return data.decode("utf-8", errors="replace")


def parse_float_attr(value: Optional[str], default: float = 50.0) -> float:
    """
    width="50", width="50px" 같은 값을 float로 파싱합니다.
    """
    if value is None:
        return default

    m = re.search(r"-?\d*\.?\d+", value)

    if not m:
        return default

    return float(m.group())


def read_svg_path_from_text(svg_text: str) -> Tuple[str, float, float]:
    """
    tar 안의 SVG text에서 path d, width, height를 읽습니다.

    반환:
    path_d, width, height
    """
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as e:
        raise ValueError(f"Invalid SVG XML: {e}")

    width = parse_float_attr(root.attrib.get("width"), default=50.0)
    height = parse_float_attr(root.attrib.get("height"), default=50.0)

    # width/height가 없으면 viewBox에서 추정
    view_box = root.attrib.get("viewBox") or root.attrib.get("viewbox")

    if view_box and (width == 50.0 or height == 50.0):
        nums = re.findall(r"-?\d*\.?\d+", view_box)

        if len(nums) == 4:
            width = float(nums[2])
            height = float(nums[3])

    path_ds = []

    for elem in root.iter():
        # namespace가 있어도 tag 끝이 path면 처리
        if elem.tag.lower().endswith("path"):
            d = elem.attrib.get("d")

            if d:
                path_ds.append(d.strip())

    if not path_ds:
        raise ValueError("No <path d='...'> found in SVG")

    return " ".join(path_ds), width, height


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
            out.append("Z")
            i += 1

        else:
            # H/V/A/S/T 등이 있다면 현재는 skip
            # 필요하면 command별 처리 추가
            i += 1

    return " ".join(out)


def skeleton_path_to_integer_points(
    path_d: str,
    scale: int = 10,
    convert_c_to_endpoint: bool = True,
    add_end: bool = True,
) -> str:
    """
    skeleton SVG path를 point sequence로 변환합니다.

    기존:
    M 17.0 27.3 L 29.1 27.3 L 34.9 26.1

    변경:
    PATH 170 273 291 273 349 261 END

    여러 subpath:
    M 36.5 17.8 L 19.6 28.3 M 21.3 29.3 L 36.7 38.9

    ->
    PATH 365 178 196 283 PATH 213 293 367 389 END
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
                "PATH",
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
                float_to_int_token(x, scale),
                float_to_int_token(y, scale),
            ])

            i += 3

        elif t == "C":
            if i + 6 >= len(toks):
                break

            if convert_c_to_endpoint:
                # C x1 y1 x2 y2 x y 에서 endpoint x y만 사용
                x = float(toks[i + 5])
                y = float(toks[i + 6])

                out.extend([
                    float_to_int_token(x, scale),
                    float_to_int_token(y, scale),
                ])

            i += 7

        elif t == "Q":
            if i + 4 >= len(toks):
                break

            # Q x1 y1 x y 에서 endpoint x y만 사용
            x = float(toks[i + 3])
            y = float(toks[i + 4])

            out.extend([
                float_to_int_token(x, scale),
                float_to_int_token(y, scale),
            ])

            i += 5

        elif t == "Z":
            # skeleton 중심선에서는 point sequence가 중요하므로 Z는 무시
            i += 1

        else:
            i += 1

    if add_end:
        out.append("END")

    return " ".join(out)


def collect_svg_members_from_tar(
    tar_path: Path,
) -> Tuple[Dict[str, tarfile.TarInfo], Dict[str, tarfile.TarInfo]]:
    """
    tar 내부에서 aug_ttf, aug_fnt SVG member를 파일명 기준으로 수집합니다.

    반환:
    outline_members: {filename.svg: TarInfo}
    skeleton_members: {filename.svg: TarInfo}
    """
    outline_members = {}
    skeleton_members = {}

    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue

            name = normalize_tar_name(member.name)

            if not is_svg_file(name):
                continue

            group = detect_group(name)

            if group is None:
                continue

            filename = get_basename(name)

            if group == "aug_ttf":
                outline_members[filename] = member
            elif group == "aug_fnt":
                skeleton_members[filename] = member

    return outline_members, skeleton_members


def build_pairs_from_tar(
    tar_path: Path,
    coord_scale: int = 10,
):
    samples = []

    with tarfile.open(tar_path, "r:*") as tf:
        members = [
            m for m in tf.getmembers()
            if m.isfile() and is_svg_file(m.name)
        ]

        outline_members = {}
        skeleton_members = {}

        for member in members:
            name = normalize_tar_name(member.name)
            group = detect_group(name)

            if group is None:
                continue

            filename = get_basename(name)

            if group == "aug_ttf":
                outline_members[filename] = member
            elif group == "aug_fnt":
                skeleton_members[filename] = member

        common_names = sorted(
            set(outline_members.keys()) & set(skeleton_members.keys())
        )

        missing_skeleton = sorted(
            set(outline_members.keys()) - set(skeleton_members.keys())
        )

        missing_outline = sorted(
            set(skeleton_members.keys()) - set(outline_members.keys())
        )

        print(f"outline svg count: {len(outline_members)}")
        print(f"skeleton svg count: {len(skeleton_members)}")
        print(f"paired svg count: {len(common_names)}")
        print(f"missing skeleton count: {len(missing_skeleton)}")
        print(f"missing outline count: {len(missing_outline)}")

        if len(common_names) == 0:
            raise RuntimeError(
                "No paired SVG files found. "
                "Check tar structure. Expected folders containing aug_ttf and aug_fnt."
            )

        for idx, filename in enumerate(common_names):
            outline_member = outline_members[filename]
            skeleton_member = skeleton_members[filename]

            try:
                outline_svg_text = read_tar_member_text(tf, outline_member)
                skeleton_svg_text = read_tar_member_text(tf, skeleton_member)

                out_d, w, h = read_svg_path_from_text(outline_svg_text)
                sk_d, _, _ = read_svg_path_from_text(skeleton_svg_text)

                outline_int = outline_path_to_integer_path(
                    out_d,
                    scale=coord_scale,
                )

                skeleton_points = skeleton_path_to_integer_points(
                    sk_d,
                    scale=coord_scale,
                    convert_c_to_endpoint=True,
                    add_end=True,
                )

                samples.append({
                    "id": Path(filename).stem,

                    "src_text": f"outline2skeleton: outline path: {outline_int}",

                    "tgt_text": f"skeleton points: {skeleton_points}",

                    "width": w,
                    "height": h,
                    "coord_scale": coord_scale,
                })

            except Exception as e:
                print(f"[SKIP] {filename}: {e}")

            if (idx + 1) % 1000 == 0:
                print(f"processed: {idx + 1}/{len(common_names)}")

    return samples


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--tar_path", required=True)
    ap.add_argument("--out_dir", default="dataset/processed_from_tar_points")
    ap.add_argument("--test_size", type=float, default=0.1)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--coord_scale", type=int, default=10)

    args = ap.parse_args()

    tar_path = Path(args.tar_path)

    rows = build_pairs_from_tar(
        tar_path=tar_path,
        coord_scale=args.coord_scale,
    )

    if len(rows) == 0:
        raise RuntimeError("No samples were created.")

    train_rows, test_rows = train_test_split(
        rows,
        test_size=args.test_size,
        random_state=42,
    )

    train_rows, val_rows = train_test_split(
        train_rows,
        test_size=args.val_size,
        random_state=42,
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