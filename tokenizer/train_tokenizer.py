from __future__ import annotations

import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

SPECIAL = ["<pad>", "<s>", "</s>", "<unk>", "<OUTLINE>", "<SKELETON>"]


def yield_tokens(jsonl_path: Path):
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield row["outline_tokens"] + row["skeleton_tokens"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", default="dataset/processed/train.jsonl")
    ap.add_argument("--out_dir", default="tokenizer/artifact")
    args = ap.parse_args()

    vocab = {t: i for i, t in enumerate(SPECIAL)}
    idx = len(vocab)
    for tokens in yield_tokens(Path(args.train_jsonl)):
        for t in tokens:
            if t not in vocab:
                vocab[t] = idx
                idx += 1

    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(out_dir / "tokenizer.json"),
        bos_token="<s>", eos_token="</s>",
        unk_token="<unk>", pad_token="<pad>",
        additional_special_tokens=["<OUTLINE>", "<SKELETON>"]
    )
    hf_tok.save_pretrained(out_dir)
    print(f"vocab={len(vocab)}")
