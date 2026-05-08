from __future__ import annotations

import math
from typing import List, Tuple
import cairosvg
from PIL import Image
import io
import numpy as np


def rasterize_svg(svg_text: str, size: int = 256) -> np.ndarray:
    png = cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), output_width=size, output_height=size)
    img = Image.open(io.BytesIO(png)).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def pixel_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def chamfer_distance(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> float:
    ap = np.argwhere(a < threshold)
    bp = np.argwhere(b < threshold)
    if len(ap) == 0 or len(bp) == 0:
        return float("inf")
    d1 = np.mean([np.min(np.sum((bp - p) ** 2, axis=1)) for p in ap])
    d2 = np.mean([np.min(np.sum((ap - p) ** 2, axis=1)) for p in bp])
    return float(math.sqrt(d1) + math.sqrt(d2))
