#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

export CUDA_VISIBLE_DEVICES=0

# python inference/generate.py \
#     --input_svg "/home/jmseo1216/deepfont/font_llm_finetuning/tmp_input.svg" \
#     --output_svg "output_aug_0051_83.svg" \
#     --adapter_dir "/home/jmseo1216/deepfont/font_llm_finetuning/checkpoints/flan_t5/checkpoint-620" \
#     --tokenizer_dir "tokenizer/artifact"

# ---------------------------------------------------------------------------------------------------------------------

# # 단일 file 버전 
# python inference/generate_raw_svg_dir.py \
#     --input_svg "//home/jmseo1216/deepfont/aug_ttf/aug_0024_66.svg" \
#     --model_dir "/home/jmseo1216/deepfont/font_llm_finetuning/checkpoints/int-flan_t5_base_1e-4_scheduler/checkpoint-7230" \
#     --output_svg "output_raw_aug.svg" 

# "/home/jmseo1216/deepfont/aug_ttf/aug_0053_64.svg"    # "@"  버전 (test.jsonl)

# # # dir 버전 
# python inference/generate_raw_svg_dir.py \
#     --input_dir "/home/jmseo1216/deepfont/test_svg" \
#     --model_dir "/home/jmseo1216/deepfont/font_llm_finetuning/checkpoints/flan_t5_base_1e-4_scheduler/checkpoint-7230" \
#     --output_dir "/home/jmseo1216/deepfont/test_svg_outputs" \
#     --max_input_length 1024 \
#     --max_new_tokens 512 \


# ---------------------------------------- dir_both_int_end------------------------------------------
# python inference/generate_raw_svg_dir_both_int_end.py \
#     --input_svg "/home/jmseo1216/deepfont/aug_ttf/aug_0013_59.svg" \
#     --model_dir "/home/jmseo1216/deepfont/font_llm_finetuning/checkpoints/e20_add_B_both_int_end-flan_t5_base_1e-4_scheduler/checkpoint-4960" \
#     --output_svg "output_raw_aug.svg"  \
#     --device "cpu"

# ---------------------------------------- dir_both_int_end_tgtseq ------------------------------------------
python inference/generate_raw_svg_dir_both_int_end_tgtseq.py \
    --input_svg "/home/jmseo1216/deepfont/test_dataset_folder/test_dataset/arial_aug_0000_65.svg" \
    --model_dir "/home/jmseo1216/deepfont/font_llm_finetuning/checkpoints/all_aug300_e30_tgtseq_both_int_end-flan_t5_large/checkpoint-21495" \
    --output_svg "output_raw_aug.svg" \
    --coord_scale 10 \
    --device "cpu"


# python inference/generate_raw_svg_dir_both_int_end_tgtseq.py \
#     --input_dir "/home/jmseo1216/deepfont/test_dataset_folder/titillium_web_prettf_svg" \
#     --model_dir "/home/jmseo1216/deepfont/font_llm_finetuning/checkpoints/int-flan_t5_base_1e-4_scheduler/checkpoint-7230" \
#     --output_dir "/home/jmseo1216/deepfont/test_dataset_folder/titillium_web_output_int-flan_t5_base_1e-4_scheduler_7230" \
#     --coord_scale 10 \
#     --device "cpu"


# --input_svg "/home/jmseo1216/deepfont/aug_ttf/aug_0024_66.svg" \
# --input_svg "/home/jmseo1216/deepfont/all_dataset/aug_ttf/arial_aug_0000_33.svg" \