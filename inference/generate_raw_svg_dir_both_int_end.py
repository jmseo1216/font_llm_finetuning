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

# output은 integer path이므로 정수만 추출
# END도 함께 감지해서 END 이후는 버림
INT_PATH_RE = re.compile(r"(END)|([MmLlZz])|(-?\d+)")


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
    inference input SVG outline path를 학습 때와 동일하게 integer path로 변환합니다.

    입력:
    M 11.6 10.1 L 11.6 42.9 C 30.4 42.9 32.9 42.6 34.8 41.9 Z

    출력:
    M 116 101 L 116 429 C 304 429 329 426 348 419 Z
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
            # H/V/A/S/T 등 현재 데이터에서 사용하지 않는 command는 skip
            # 필요하면 추후 command별 처리 추가
            i += 1

    return " ".join(out)


def clean_generated_integer_skeleton(text: str) -> str:
    """
    모델 출력에서 'skeleton path:' 뒤의 M/L/Z + integer token만 추출합니다.
    END가 나오면 그 이후는 버립니다.

    예:
    <pad> skeleton path: M 171 276 L 286 276 END</s>
    ->
    M 171 276 L 286 276
    """
    text = text.replace("<pad>", " ")
    text = text.replace("</s>", " ")
    text = text.replace("<s>", " ")

    lower_text = text.lower()
    key = "skeleton path:"

    if key in lower_text:
        start = lower_text.index(key) + len(key)
        text = text[start:]

    toks = []

    for m in INT_PATH_RE.finditer(text):
        end_token, cmd, num = m.groups()

        if end_token == "END":
            break

        if cmd:
            toks.append(cmd.upper())
        elif num:
            toks.append(num)

    return " ".join(toks)


def integer_skeleton_to_svg_path(
    integer_path: str,
    scale: int = 10,
) -> str:
    """
    integer skeleton path를 실제 SVG path로 복원합니다.

    입력:
    M 171 276 L 286 276 L 389 236

    출력:
    M 17.1 27.6 L 28.6 27.6 L 38.9 23.6
    """
    toks = integer_path.split()

    out = []
    i = 0

    while i < len(toks):
        t = toks[i]

        if t in ["M", "L"]:
            if i + 2 >= len(toks):
                break

            try:
                x = int(toks[i + 1]) / scale
                y = int(toks[i + 2]) / scale
            except ValueError:
                break

            out.extend([
                t,
                f"{x:.1f}",
                f"{y:.1f}",
            ])

            i += 3

        elif t == "Z":
            out.append("Z")
            i += 1

        else:
            i += 1

    return " ".join(out)


def load_train_meta(model_dir: Path):
    meta_path = model_dir / "train_meta.json"

    if not meta_path.exists():
        return {}

    return json.loads(
        meta_path.read_text(encoding="utf-8")
    )


def generate_one_svg(
    input_svg: Path,
    output_svg: Path,
    tokenizer,
    model,
    device: str,
    max_input_length: int = 1024,
    max_new_tokens: int = 512,
    coord_scale: int = 10,
):
    # =====================================================
    # 1. input SVG 읽기
    # =====================================================
    path_d, w, h = read_svg_path(input_svg)

    print("=" * 80)
    print(f"[INPUT] {input_svg}")
    print(f"input path length: {len(path_d)}")

    # =====================================================
    # 2. 학습 때와 동일하게 input outline을 integer path로 변환
    # =====================================================
    outline_int = outline_path_to_integer_path(
        path_d,
        scale=coord_scale,
    )

    print(f"integer input path length: {len(outline_int)}")

    # 학습 JSONL과 반드시 동일한 prefix 사용
    src = f"outline2skeleton: outline path: {outline_int}"

    print("\n[SRC]")
    print(src[:500])

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
    print(out_text[:1000])

    print("\n[GEN TOKEN DEBUG]")
    print("generated token length:", gen.shape[-1])
    print("eos count:", (gen[0] == tokenizer.eos_token_id).sum().item())
    print("last token id:", gen[0, -1].item())
    print("eos token id:", tokenizer.eos_token_id)

    # =====================================================
    # 4. integer skeleton 추출 후 SVG path로 복원
    # =====================================================
    gen_integer_skeleton = clean_generated_integer_skeleton(out_text)

    print("\n[CLEANED INTEGER SKELETON]")
    print(gen_integer_skeleton[:1000])

    gen_path_d = integer_skeleton_to_svg_path(
        gen_integer_skeleton,
        scale=coord_scale,
    )

    print("\n[CLEANED SVG PATH]")
    print(gen_path_d[:1000])
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
    ap.add_argument("--model_dir", default="checkpoints/flan_t5_raw_full", help="Full fine-tuned model directory")
    ap.add_argument("--output_svg", default=None, help="Output SVG path for single-file inference")
    ap.add_argument("--output_dir", default=None, help="Output directory for batch inference")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_input_length", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    # build_dataset.py에서 사용한 coord_scale과 반드시 동일해야 함
    ap.add_argument("--coord_scale", type=int, default=10)

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
        "unknown"
    )

    print("model_dir:", model_dir)
    print("base_model:", base_model_name)
    print("device:", args.device)
    print("coord_scale:", args.coord_scale)

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
            output_svg = Path(f"inf_{input_svg.name}")
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
            coord_scale=args.coord_scale,
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
                coord_scale=args.coord_scale,
            )

        print("=" * 80)
        print(f"Batch inference done. Saved to: {output_dir}")