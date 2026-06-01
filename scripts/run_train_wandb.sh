#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_PROJECT="deepfont-skeleton-generation"
export WANDB_RUN_NAME="flan-t5-large-all-points-lr3e-5-bs32"
export WANDB_LOG_MODEL="false"
export WANDB_MODE=offline

proxychains python training/train_raw_svg.py \
    --model_name google/flan-t5-large \
    --train_jsonl dataset/processed_v2_all_aug300_both_int_end_tgtseq/train.jsonl \
    --val_jsonl dataset/processed_v2_all_aug300_both_int_end_tgtseq/val.jsonl \
    --out_dir checkpoints/all_v2_aug300_e30_tgtseq_both_int_end-flan_t5_large