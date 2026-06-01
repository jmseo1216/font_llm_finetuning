export PYTHONPATH=$PYTHONPATH:$(pwd)

export CUDA_VISIBLE_DEVICES=0

# python training/train_seq2seq.py \
#   --model_name google/flan-t5-small \
#   --tokenizer_dir tokenizer/artifact \
#   --train_jsonl dataset/processed/train.jsonl \
#   --val_jsonl dataset/processed/val.jsonl \
#   --out_dir checkpoints/flan_t5


## quantize_coordinates 사용하지 않는 버전 
python training/train_raw_svg.py \
    --model_name google/flan-t5-large \
    --train_jsonl dataset/processed_all_aug300_both_int_end_tgtseq/train.jsonl \
    --val_jsonl dataset/processed_all_aug300_both_int_end_tgtseq/val.jsonl \
    --out_dir checkpoints/all_aug300_e30_tgtseq_both_int_end-flan_t5_large