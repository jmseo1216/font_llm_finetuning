python training/train_seq2seq.py \
  --model_name google/byt5-small \
  --tokenizer_dir tokenizer/artifact \
  --train_jsonl dataset/processed/train.jsonl \
  --val_jsonl dataset/processed/val.jsonl \
  --out_dir checkpoints/byt5_lora
