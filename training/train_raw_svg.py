from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import torch
from datasets import Dataset as HFDataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from dataset.svg_seq2seq_dataset import SVGSeq2SeqDataset


def to_hf(ds, tokenizer, max_src=1024, max_tgt=1024):
    rows = [ds[i] for i in range(len(ds))]
    hfd = HFDataset.from_list(rows)

    def preprocess(ex):
        x = tokenizer(
            ex["src_text"],
            truncation=True,
            max_length=max_src,
        )

        y = tokenizer(
            text_target=ex["tgt_text"],
            truncation=True,
            max_length=max_tgt,
        )

        labels = [
            token if token != tokenizer.pad_token_id else -100
            for token in y["input_ids"]
        ]

        x["labels"] = labels
        return x

    return hfd.map(
        preprocess,
        remove_columns=hfd.column_names,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name", default="google/flan-t5-small")
    ap.add_argument("--train_jsonl", default="dataset/processed_raw/train.jsonl")
    ap.add_argument("--val_jsonl", default="dataset/processed_raw/val.jsonl")
    ap.add_argument("--out_dir", default="checkpoints/flan_t5_raw_full")
    args = ap.parse_args()

    print(f"model name: {args.model_name}")

    # =====================================================
    # tokenizer: FLAN-T5 원래 tokenizer 그대로 사용
    # =====================================================
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("tokenizer len:", len(tokenizer))
    print("pad id:", tokenizer.pad_token_id)
    print("eos id:", tokenizer.eos_token_id)
    print("unk id:", tokenizer.unk_token_id)

    # =====================================================
    # model: resize_token_embeddings 하지 않음
    # =====================================================
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    model.config.decoder_start_token_id = tokenizer.pad_token_id

    train_ds = to_hf(
        SVGSeq2SeqDataset(args.train_jsonl),
        tokenizer,
        max_src=1024,
        max_tgt=1024,
    )

    val_ds = to_hf(
        SVGSeq2SeqDataset(args.val_jsonl),
        tokenizer,
        max_src=1024,
        max_tgt=1024,
    )

    # =====================================================
    # dataset debug
    # =====================================================
    sample = train_ds[0]    

    print("\n[DATASET DEBUG]")
    print(sample.keys())
    print("input_ids[:50]:", sample["input_ids"][:50])
    print("labels[:50]:", sample["labels"][:50])
    print("input len:", len(sample["input_ids"]))
    print("label len:", len(sample["labels"]))

    print("\n[DECODE DEBUG]")
    print(tokenizer.decode(sample["input_ids"][:120]))
    print(tokenizer.decode([x for x in sample["labels"][:120] if x != -100]))

    unk_id = tokenizer.unk_token_id

    if unk_id is not None:
        print("\n[UNK CHECK]")
        print("unk id:", unk_id)
        print("input unk count:", sum(1 for x in sample["input_ids"] if x == unk_id))
        print("label unk count:", sum(1 for x in sample["labels"] if x == unk_id))

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    # =====================================================
    # manual loss debug
    # =====================================================
    batch = collator([train_ds[0], train_ds[1]])

    batch = {
        k: v.to(model.device)
        for k, v in batch.items()
    }

    model.eval()

    with torch.no_grad():
        outputs = model(**batch)

    print("\n[MANUAL LOSS DEBUG]")
    print("manual loss:", outputs.loss.item())
    print("input_ids shape:", batch["input_ids"].shape)
    print("labels shape:", batch["labels"].shape)
    print("max input id:", batch["input_ids"].max().item())
    print("max label id:", batch["labels"][batch["labels"] != -100].max().item())
    print("tokenizer len:", len(tokenizer))
    print("model vocab size:", model.config.vocab_size)
    print("embedding shape:", model.get_input_embeddings().weight.shape)
    print("=" * 80)

    model.train()

    # train_args = Seq2SeqTrainingArguments(
    #     output_dir=args.out_dir,
    #     per_device_train_batch_size=4,
    #     per_device_eval_batch_size=4,
    #     gradient_accumulation_steps=8,
    #     learning_rate=5e-5,   
    #     lr_scheduler_type="constant_with_warmup",
    #     warmup_ratio=0.03,
    #     num_train_epochs=20,
    #     bf16=False,
    #     max_grad_norm=1.0,
    #     logging_steps=2,
    #     eval_strategy="steps",
    #     eval_steps=60,
    #     save_steps=240,
    #     save_total_limit=10,
    #     predict_with_generate=True,
    #     generation_max_length=512,

    #     report_to="none",
    # )

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,

        learning_rate=3e-5,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.03,

        num_train_epochs=30,

        bf16=False,
        max_grad_norm=1.0,

        logging_steps=100,

        eval_strategy="steps",
        eval_steps=4299,

        save_steps=4299,
        save_total_limit=10,

        predict_with_generate=False,
        generation_max_length=512,

        report_to="wandb",
        run_name=os.environ.get(
            "WANDB_RUN_NAME",
            "flan-t5-large"  # wnadb_run_name 없으면 기본값 사용 
        ),
    )
    
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

    meta = {
        "base_model": args.model_name,
        "tokenizer_vocab_size": len(tokenizer),
        "format": "raw_svg_text_no_added_tokens",
    }

    Path(args.out_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    (Path(args.out_dir) / "train_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8"
    )

    print(f"Saved model to {args.out_dir}")