from __future__ import annotations

import argparse
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast
from peft import PeftModel
from src.font_pipeline.svg_utils import read_svg_path, tokenize_path, quantize_coordinates, dequantize_sequence, tokens_to_svg_path

TEMPLATE = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><path d="{d}" fill="none" stroke="black"/></svg>'


def clean_tokens(tokens):
    return [t for t in tokens if t.startswith("<CMD_") or t.startswith("<NUM_")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_svg", required=True)
    ap.add_argument("--base_model", default="google/byt5-small")
    ap.add_argument("--adapter_dir", default="checkpoints/byt5_lora")
    ap.add_argument("--tokenizer_dir", default="tokenizer/artifact")
    ap.add_argument("--output_svg", default="generated.svg")
    args = ap.parse_args()

    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(model, args.adapter_dir)

    path_d, w, h = read_svg_path(Path(args.input_svg))
    src = "<OUTLINE> " + " ".join(quantize_coordinates(tokenize_path(path_d), w, h))
    enc = tok(src, return_tensors="pt")
    gen = model.generate(**enc, max_new_tokens=1024)
    out_text = tok.decode(gen[0], skip_special_tokens=False)
    toks = clean_tokens(out_text.split())
    path_tokens = dequantize_sequence(toks, w, h)
    svg = TEMPLATE.format(w=w, h=h, d=tokens_to_svg_path(path_tokens))
    Path(args.output_svg).write_text(svg, encoding="utf-8")
    print(args.output_svg)
