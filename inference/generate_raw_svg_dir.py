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
    기준으로 path 부분만 추출.
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

    # 혹시 이전 형식이 섞인 경우
    elif "<SKELETON>" in text:
        text = text.split("<SKELETON>", 1)[1]

    # 불필요한 토큰 제거
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


def generate_one_svg(
    input_svg: Path,
    output_svg: Path,
    tokenizer,
    model,
    device: str,
    max_input_length: int = 1024,
    max_new_tokens: int = 512,
):
    # =====================================================
    # 1. input SVG 읽기
    # =====================================================
    path_d, w, h = read_svg_path(input_svg)

    print("=" * 80)
    print(f"[INPUT] {input_svg}")
    print(f"input path length: {len(path_d)}")
    
    # =====================================================
    # 2. 학습 때와 동일한 src_text 형식
    # =====================================================
    src = f"outline2skeleton: outline path: {path_d}"

    enc = tokenizer(
        src,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(device)

    print("input_ids shape:", enc["input_ids"].shape)

    # =====================================================
    # 3. generate
    # =====================================================
    with torch.no_grad():
        gen = model.generate(
            **enc,

            max_new_tokens=max_new_tokens,
            # min_new_tokens=200,
            do_sample=False,
            num_beams=4,

            repetition_penalty=1.3,
            no_repeat_ngram_size=3,

            early_stopping=True,

            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.pad_token_id,
        )

    out_text = tokenizer.decode(
        gen[0],
        skip_special_tokens=False
    )

    print("\n[RAW OUTPUT]")
    print(out_text[:500])

    # =====================================================
    # 4. path만 추출
    # =====================================================
    gen_path_d = clean_generated_svg_path(out_text)

    print("\n[CLEANED PATH]")
    print(gen_path_d[:500])
    print("cleaned path length:", len(gen_path_d))

    # =====================================================
    # 5. SVG 저장
    # =====================================================
    svg = TEMPLATE.format(
        w=w,
        h=h,
        d=gen_path_d
    )

    output_svg.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_svg.write_text(
        svg,
        encoding="utf-8"
    )

    print(f"[SAVED] {output_svg}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # 단일 파일 추론
    ap.add_argument("--input_svg", default=None, help="Single outline SVG file path")
    # 디렉토리 일괄 추론
    ap.add_argument("--input_dir", default=None, help="Directory containing input SVG files")
    ap.add_argument("--model_dir", default="checkpoints/flan_t5_raw_full", help="Full fine-tuned model directory" )
    ap.add_argument("--output_svg", default=None, help="Output SVG path for single-file inference")
    ap.add_argument("--output_dir", default=None, help="Output directory for batch inference")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_input_length", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    # =====================================================
    # 입력 조건 체크
    # =====================================================
    if args.input_svg is None and args.input_dir is None:
        raise ValueError(
            "Either --input_svg or --input_dir must be provided."
        )

    if args.input_svg is not None and args.input_dir is not None:
        raise ValueError(
            "Use only one of --input_svg or --input_dir, not both."
        )

    model_dir = Path(args.model_dir)

    meta = load_train_meta(model_dir)
    base_model_name = meta.get(
        "base_model",
        "google/flan-t5-small"
    )

    print("model_dir:", model_dir)
    print("base_model:", base_model_name)
    print("device:", args.device)

    # =====================================================
    # 학습된 tokenizer/model 로드
    # =====================================================
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    model.config.decoder_start_token_id = tokenizer.pad_token_id

    model.to(args.device)
    model.eval()

    # =====================================================
    # 단일 SVG 추론
    # =====================================================
    if args.input_svg is not None:
        input_svg = Path(args.input_svg)

        if args.output_svg is None:
            output_svg = Path("generated.svg")
        else:
            output_svg = Path(args.output_svg)

        generate_one_svg(
            input_svg=input_svg,
            output_svg=output_svg,
            tokenizer=tokenizer,
            model=model,
            device=args.device,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
        )

    # =====================================================
    # 디렉토리 일괄 추론
    # =====================================================
    else:
        input_dir = Path(args.input_dir)

        if args.output_dir is None:
            output_dir = Path("generated_outputs")
        else:
            output_dir = Path(args.output_dir)

        svg_files = sorted(input_dir.glob("*.svg"))

        if len(svg_files) == 0:
            raise FileNotFoundError(
                f"No SVG files found in {input_dir}"
            )

        print(f"Found {len(svg_files)} SVG files.")

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for input_svg in svg_files:
            output_svg = output_dir / f"inf_{input_svg.name}"

            generate_one_svg(
                input_svg=input_svg,
                output_svg=output_svg,
                tokenizer=tokenizer,
                model=model,
                device=args.device,
                max_input_length=args.max_input_length,
                max_new_tokens=args.max_new_tokens,
            )

        print("=" * 80)
        print(f"Batch inference done. Saved to: {output_dir}")