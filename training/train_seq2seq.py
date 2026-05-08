from __future__ import annotations

import argparse
import numpy as np
from datasets import Dataset as HFDataset
from transformers import (
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
        x["labels"] = y["input_ids"]
        return x

    return hfd.map(preprocess, remove_columns=hfd.column_names)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="google/byt5-small")
    ap.add_argument("--tokenizer_dir", default="tokenizer/artifact")
    ap.add_argument("--train_jsonl", default="dataset/processed/train.jsonl")
    ap.add_argument("--val_jsonl", default="dataset/processed/val.jsonl")
    ap.add_argument("--out_dir", default="checkpoints/byt5_lora")
    args = ap.parse_args()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    peft_cfg = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_cfg)

    train_ds = to_hf(SVGSeq2SeqDataset(args.train_jsonl), tokenizer)
    val_ds = to_hf(SVGSeq2SeqDataset(args.val_jsonl), tokenizer)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=10,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        predict_with_generate=True,
        generation_max_length=1024,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
