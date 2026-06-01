from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast

from src.font_pipeline.svg_utils import (
    dequantize_sequence,
    quantize_coordinates,
    read_svg_path,
    tokenize_path,
    tokens_to_svg_path,
)

TEMPLATE = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><path d="{d}" fill="none" stroke="black"/></svg>'


def clean_tokens(tokens):
    return [t for t in tokens if t.startswith("<CMD_") or t.startswith("<NUM_")]


def load_train_meta(adapter_dir: Path):
    meta_path = adapter_dir / "train_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_svg", required=True, help="Outline SVG file path")
    ap.add_argument("--base_model", default=None, help="Optional override for base model")
    ap.add_argument("--adapter_dir", default="checkpoints/byt5_lora")
    ap.add_argument("--tokenizer_dir", default="tokenizer/artifact")
    ap.add_argument("--output_svg", default="generated.svg")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    meta = load_train_meta(adapter_dir)
    base_model_name = args.base_model or meta.get("base_model", "google/flan-t5-small")

    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    print(f"tok: {tok}")
    print(f'test:{tok.tokenize("<CMD_M> <NUM_168>")}')
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    # model.config.decoder_start_token_id = tok.pad_token_id  # gpt 보고 추가한 것 
    
    model.resize_token_embeddings(len(tok))
    model.config.tie_word_embeddings = False
    # model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to(args.device)
    model.eval()

    path_d, w, h = read_svg_path(Path(args.input_svg))
    print(f"path_d: {path_d }")
    print(f"length_path_d: {len(path_d)}")
    src = "<OUTLINE> " + " ".join(quantize_coordinates(tokenize_path(path_d), w, h, bins=256))

    print(f"src: {src}")
    enc = tok(src, return_tensors="pt").to(args.device)
    print(f"enc: {enc}")
    with torch.no_grad():
        # gen = model.generate(**enc, max_new_tokens=1024)
        # gen = model.generate(
        #     **enc,
        #     max_new_tokens=256,
        #     do_sample=True,
        #     top_k=20,
        #     top_p=0.9,
        #     temperature=0.8,
        #     repetition_penalty=1.2,
        #     eos_token_id=tok.eos_token_id,
        #     pad_token_id=tok.pad_token_id,
        # )
        # gpt가 알려준거 
        gen = model.generate(
            **enc,
            max_new_tokens=256,
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
        )

    out_text = tok.decode(gen[0], skip_special_tokens=False)
    print(f"out_test: {out_text[:15]}")
    toks = clean_tokens(out_text.split())
    print(f"toks: {toks}")
    print(f"len toks: {len(toks)}")
    path_tokens = dequantize_sequence(toks, w, h,bins=256)
    svg = TEMPLATE.format(w=w, h=h, d=tokens_to_svg_path(path_tokens))
    Path(args.output_svg).write_text(svg, encoding="utf-8")
    print(args.output_svg)