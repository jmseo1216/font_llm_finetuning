from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from dataset.svg_seq2seq_dataset import SVGSeq2SeqDataset


def to_hf(ds, tokenizer, max_src=1024, max_tgt=1024):
    rows = [ds[i] for i in range(len(ds))]
    hfd = HFDataset.from_list(rows)

    def preprocess(ex):
        x = tokenizer(ex["src_text"], truncation=True, max_length=max_src)
        y = tokenizer(text_target=ex["tgt_text"], truncation=True, max_length=max_tgt)
        labels = [
            (t if t != tokenizer.pad_token_id else -100)
            for t in y["input_ids"]
        ]
        x["labels"] = labels
        return x

    return hfd.map(preprocess, remove_columns=hfd.column_names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # ap.add_argument("--model_name", default="google/byt5-small")
    ap.add_argument("--model_name", default="google/flan-t5-small")
    ap.add_argument("--tokenizer_dir", default="tokenizer/artifact")
    ap.add_argument("--train_jsonl", default="dataset/processed/train.jsonl")
    ap.add_argument("--val_jsonl", default="dataset/processed/val.jsonl")
    # ap.add_argument("--out_dir", default="checkpoints/byt5_lora")
    ap.add_argument("--out_dir", default="checkpoints/flan_t5_lora")
    args = ap.parse_args()

    print(f"model name: {args.model_name}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # peft_cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q", "k", "v", "o"])
    # model = get_peft_model(model, peft_cfg)

    train_ds = to_hf(SVGSeq2SeqDataset(args.train_jsonl), tokenizer)
    val_ds = to_hf(SVGSeq2SeqDataset(args.val_jsonl), tokenizer)


    # =========================
    # dataset debug

    # =========================
    # sample = train_ds[0]

    # print("\n[DATASET DEBUG]")
    # print(sample.keys())

    # print("\ninput_ids[:50]")
    # print(sample["input_ids"][:50])

    # print("\nlabels[:50]")
    # print(sample["labels"][:50])

    # print("\ninput len:", len(sample["input_ids"]))
    # print("label len:", len(sample["labels"]))

    # # unk 검사
    # unk_id = tokenizer.unk_token_id

    # input_unk_count = sum(1 for x in sample["input_ids"] if x == unk_id)
    # label_unk_count = sum(1 for x in sample["labels"] if x == unk_id)

    # print("\nUNK CHECK")
    # print("unk token id:", unk_id)
    # print("input unk count:", input_unk_count)
    # print("label unk count:", label_unk_count)

    # # decode 검사
    # print("\nDECODE CHECK")
    # print(tokenizer.decode(sample["input_ids"][:100]))
    # print(tokenizer.decode([x for x in sample["labels"][:100] if x != -100]))

    # print("=" * 80)
    # print(sample["labels"])
    # print(sum(1 for x in sample["labels"] if x == -100))

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


    train_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,  # 2e-4
        num_train_epochs=20,
        bf16=True,
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=60,
        save_steps=240,
        predict_with_generate=True,
        generation_max_length=1024,
        report_to="none",
    )

    # # 2. trainer.train() 실행 직전에 종료 코드 삽입
    # import sys
    # print("\n[DEBUG COMPLETE] 학습을 시작하지 않고 종료합니다.")
    # sys.exit(0)

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    meta = {"base_model": args.model_name, "tokenizer_vocab_size": len(tokenizer)}
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
