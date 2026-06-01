from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET

COMMAND_RE = re.compile(r"([MmLlHhVvCcQqSsTtAaZz])|(-?\d*\.?\d+(?:e[-+]?\d+)?)")


@dataclass
class SVGSample:
    glyph_id: str
    outline_tokens: List[str]
    skeleton_tokens: List[str]
    width: float
    height: float


def read_svg_path(svg_path: Path) -> tuple[str, float, float]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    view_box = root.attrib.get("viewBox", "0 0 1024 1024").split()
    width = float(view_box[2])
    height = float(view_box[3])
    paths = []
    for elem in root.iter():
        if elem.tag.endswith("path") and "d" in elem.attrib:
            paths.append(elem.attrib["d"])
    if not paths:
        raise ValueError(f"No path in {svg_path}")
    return " ".join(paths), width, height


def tokenize_path(path_d: str) -> List[str]:
    toks = []
    for m in COMMAND_RE.finditer(path_d):
        cmd, num = m.groups()
        if cmd:
            toks.append(cmd.upper())
        else:
            toks.append(num)
    return toks


# def quantize_coordinates(tokens: List[str], width: float, height: float, bins: int = 256) -> List[str]:
#     out = []
#     max_dim = max(width, height, 1.0)
#     for t in tokens:
#         if re.fullmatch(r"-?\d*\.?\d+(?:e[-+]?\d+)?", t):
#             v = float(t)
#             n = (v / max_dim + 1.0) / 2.0
#             q = min(bins - 1, max(0, int(round(n * (bins - 1)))))
#             out.append(f"<NUM_{q}>")
#         else:
#             out.append(f"<CMD_{t}>")
#     return out


# def dequantize_sequence(tokens: List[str], width: float, height: float, bins: int = 256) -> List[str]:
#     out = []
#     max_dim = max(width, height, 1.0)
#     for t in tokens:
#         if t.startswith("<NUM_"):
#             q = int(t[5:-1])
#             n = q / (bins - 1)
#             v = (n * 2.0 - 1.0) * max_dim
#             out.append(f"{v:.2f}")
#         elif t.startswith("<CMD_"):
#             out.append(t[5:-1])
#     return out




def quantize_coordinates(
    tokens: List[str],
    width: float,
    height: float,
    bins: int = 256
) -> List[str]:

    out = []

    prev_x = 0.0
    prev_y = 0.0

    coord_idx = 0

    REL_SCALE = 10.0

    for t in tokens:

        if re.fullmatch(r"-?\d*\.?\d+(?:e[-+]?\d+)?", t):

            v = float(t)

            # x
            if coord_idx % 2 == 0:
                delta = v - prev_x
                prev_x = v

            # y
            else:
                delta = v - prev_y
                prev_y = v

            coord_idx += 1

            # normalize to [0,1]
            n = (delta / REL_SCALE + 1.0) / 2.0

            q = min(
                bins - 1,
                max(0, int(round(n * (bins - 1))))
            )

            out.append(f"<NUM_{q}>")

        else:

            out.append(f"<CMD_{t}>")

            if t.upper() == "M":
                prev_x = 0.0
                prev_y = 0.0
                coord_idx = 0

    return out



def dequantize_sequence(
    tokens: List[str],
    width: float,
    height: float,
    bins: int = 256
) -> List[str]:

    out = []

    prev_x = 0.0
    prev_y = 0.0

    coord_idx = 0

    REL_SCALE = 10.0

    for t in tokens:

        if t.startswith("<NUM_"):

            q = int(t[5:-1])

            n = q / (bins - 1)

            # 핵심 수정
            delta = (n * 2.0 - 1.0) * REL_SCALE

            # x
            if coord_idx % 2 == 0:
                v = prev_x + delta
                prev_x = v

            # y
            else:
                v = prev_y + delta
                prev_y = v

            coord_idx += 1

            out.append(f"{v:.2f}")

        elif t.startswith("<CMD_"):

            cmd = t[5:-1]

            out.append(cmd)

            if cmd.upper() == "M":
                prev_x = 0.0
                prev_y = 0.0
                coord_idx = 0

    return out




def tokens_to_svg_path(tokens: List[str]) -> str:
    return " ".join(tokens)
