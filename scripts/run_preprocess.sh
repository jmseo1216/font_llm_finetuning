export PYTHONPATH=$PYTHONPATH:$(pwd)

# python preprocess/build_dataset.py \
#   --outline_dir "/home/jmseo1216/deepfont/aug_fnt" \
#   --skeleton_dir "/home/jmseo1216/deepfont/aug_ttf" \
#   --out_dir dataset/processed --bins 256


## quantize_coordinates 사용하지 않는 버전 
# python preprocess/build_dataset.py \
#   --outline_dir "/home/jmseo1216/deepfont/aug_ttf" \
#   --skeleton_dir "/home/jmseo1216/deepfont/aug_fnt" \
#   --out_dir dataset/processed 


## source내뚜고 target만 int 로 만드는 버전 
# python preprocess/build_dataset_target_int.py \
#   --outline_dir "/home/jmseo1216/deepfont/aug_ttf" \
#   --skeleton_dir "/home/jmseo1216/deepfont/aug_fnt" \
#   --out_dir "dataset/processed_int" \
#   --coord_scale 10


## (source, target) int 로 만드는 버전  && target 마지막에 END 추가 버전 
# python preprocess/build_dataset_src_tgt_int.py \
#   --outline_dir "/home/jmseo1216/deepfont/aug_ttf" \
#   --skeleton_dir "/home/jmseo1216/deepfont/aug_fnt" \
#   --out_dir "dataset/processed_both_int_end" \
#   --coord_scale 10

## (source, target) int 로 만드는 버전  && target 마지막에 END 추가 버전 && target sequence 버전 
python preprocess/build_dataset_src_tgt_int_tgtseq.py \
  --outline_dir "/home/jmseo1216/deepfont/v2_all_dataset_aug300/aug_ttf" \
  --skeleton_dir "/home/jmseo1216/deepfont/v2_all_dataset_aug300/aug_fnt" \
  --out_dir "dataset/processed_v2_all_aug300_both_int_end_tgtseq" \
  --coord_scale 10

