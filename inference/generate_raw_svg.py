from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from src.font_pipeline.svg_utils import read_svg_path


TEMPLATE = '''<svg xmlns="http://www.w3.org/2000/svg"
width="{w}"
height="{h}"
viewBox="0 0 {w} {h}">

<path d="{d}"
fill="none"
stroke="black"
stroke-width="1"/>

</svg>'''


COMMAND_RE = re.compile(
    r"([MmLlHhVvCcQqSsTtAaZz])|(-?\d*\.?\d+(?:e[-+]?\d+)?)"
)


def load_train_meta(model_dir: Path):
    meta_path = model_dir / "train_meta.json"

    if not meta_path.exists():
        return {}

    return json.loads(
        meta_path.read_text(encoding="utf-8")
    )


def clean_generated_svg_path(text: str) -> str:
    """
    모델 출력에서 실제 SVG path에 필요한 command/number만 남김.
    raw SVG 학습 형식:
    skeleton path: M ...
    기준으로 path 부분만 추출
    """

    # special token 제거
    text = text.replace("<pad>", " ")
    text = text.replace("</s>", " ")
    text = text.replace("<s>", " ")

    # raw SVG 방식: "skeleton path:" 뒤만 사용
    lower_text = text.lower()

    key = "skeleton path:"

    if key in lower_text:
        start = lower_text.index(key) + len(key)
        text = text[start:]

    # 혹시 예전 형식이 섞인 경우
    elif "<SKELETON>" in text:
        text = text.split("<SKELETON>", 1)[1]

    # outline 관련 문구 제거
    text = text.replace("<OUTLINE>", " ")
    text = text.replace("<SKELETON>", " ")

    # SVG path command / number만 추출
    toks = []

    for m in COMMAND_RE.finditer(text):
        cmd, num = m.groups()

        if cmd:
            toks.append(cmd.upper())
        elif num:
            toks.append(num)

    return " ".join(toks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_svg", required=True, help="Outline SVG file path")
    ap.add_argument("--model_dir", default="checkpoints/flan_t5_raw_full", help="Full fine-tuned model directory")
    ap.add_argument("--base_model", default=None, help="Optional override for base model")
    ap.add_argument("--output_svg", default="generated.svg")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)

    meta = load_train_meta(model_dir)

    base_model_name = args.base_model or meta.get(
        "base_model",
        "google/flan-t5-small"
    )

    print("model_dir:", model_dir)
    print("base_model:", base_model_name)

    # =====================================================
    # 1. 학습된 tokenizer/model 그대로 로드
    # =====================================================
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    model.config.decoder_start_token_id = tokenizer.pad_token_id

    model.to(args.device)
    model.eval()

    # =====================================================
    # 2. input SVG 읽기
    # =====================================================
    path_d, w, h = read_svg_path(Path(args.input_svg))

    print("input path_d:", path_d)
    print("input path length:", len(path_d))

    # =====================================================
    # 3. raw SVG text source
    # 학습 때 사용한 src_text 형식과 반드시 동일해야 함
    # =====================================================
    src = f"outline2skeleton: outline path: {path_d}"

    print("src:", src[:500])

    enc = tokenizer(
        src,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(args.device)

    print("input_ids shape:", enc["input_ids"].shape)

    # =====================================================
    # 4. generate
    # =====================================================
    with torch.no_grad():
        gen = model.generate(
            **enc,

            max_new_tokens=512,
            # min_new_tokens=128,

            do_sample=False,
            num_beams=4,

            repetition_penalty=1.3,
            no_repeat_ngram_size=3,

            early_stopping=True,

            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.pad_token_id,
        )

    print(f"gen: {gen}")
    
    out_text = tokenizer.decode(
        gen[0],
        skip_special_tokens=False
    )

    print("\n[RAW OUTPUT]")
    print(out_text[:1000])

    # =====================================================
    # 5. path만 추출
    # =====================================================
    gen_path_d = clean_generated_svg_path(out_text)

    print("\n[CLEANED PATH]")
    print(gen_path_d)
    print("cleaned path length:", len(gen_path_d))

    # 비어 있으면 최소한 빈 path 저장
    svg = TEMPLATE.format(
        w=w,
        h=h,
        d=gen_path_d
    )

    Path(args.output_svg).write_text(
        svg,
        encoding="utf-8"
    )

    print("saved:", args.output_svg)