# font_llm_finetuning

## 1) 논문 접근 vs 본 프로젝트 개선
- ACCV 2024 논문은 LLM을 문자 시퀀스로 간주해 outline→skeleton 매핑을 학습한다.
- 하지만 raw SVG XML 전체를 쓰면 스타일/메타데이터 잡음 때문에 sequence length가 과도하게 길어진다.
- 본 구현은 **path 중심 토큰화 + 좌표 양자화**를 적용해 구조 보존, 길이 단축, invalid 출력 감소를 노린다.
- 또한 로컬 4080 SUPER(16GB) 환경을 고려하여 GPT류 full finetune 대신 **ByT5-small + LoRA**를 기본으로 채택했다.

## 2) 아키텍처 선택
- Causal LM: 출력 제약이 약해 invalid SVG 위험이 큼.
- Instruction tuning: 유연하지만 geometry fidelity 측면에서 불리.
- **Seq2Seq(encoder-decoder)**: 입력 outline를 인코딩하고 skeleton만 디코딩하므로 과업 정합성이 높음.

기본 권장:
1. PoC: `google/byt5-small + LoRA`
2. 성능 상향: `google/byt5-base + QLoRA(4bit)` 또는 `flan-t5-base + custom vocab`

## 3) 프로젝트 구조

```
preprocess/      # paired svg -> normalized tokens -> jsonl
dataset/         # torch dataset
training/        # hf trainer + lora
tokenizer/       # wordlevel tokenizer build
evaluation/      # raster/chamfer metric
inference/       # svg -> skeleton svg
configs/         # yaml config
scripts/         # run scripts
src/font_pipeline/
```

## 4) 실행 순서

### Step 1. 전처리
```bash
bash scripts/run_preprocess.sh
```

### Step 2. 토크나이저 학습
```bash
bash scripts/run_tokenizer.sh
```

### Step 3. 학습
```bash
bash scripts/run_train.sh
```

### Step 4. 추론
```bash
python inference/generate.py --input_svg path/to/outline.svg --output_svg out.svg
```

## 5) 전처리 설계 이유
- XML 전체 대신 `<CMD_X>, <NUM_i>` 토큰 사용: 불필요한 속성 제거, sequence 단축.
- 좌표를 `NUM_0..NUM_255`로 양자화: 숫자 fragmentation 완화, tokenizer 안정화.
- 동일 파일명 pair만 사용: supervised alignment 보장.

## 6) 4080 SUPER 실전 설정
- batch size 1 + grad accumulation 16
- bf16 mixed precision
- LoRA rank 16
- max length 1024 (데이터에 맞게 512~1536 조정)

## 7) 평가
- Raster pixel L1
- Chamfer distance (이진화 포인트 간)
- SVG 유효성(파서 통과 여부)

## 8) 후속 개선 권장
- path segment-level positional token 추가
- topology constraint decoding
- invalid path repair rule 기반 후처리
- command-aware loss weighting (M/L/C/Z)

## Inference 입력 설명 (중요)
- `train.jsonl/val.jsonl/test.jsonl`은 **학습용 전처리 산출물**입니다.
- `inference/generate.py`의 `--input_svg`는 **실제 outline SVG 파일 경로**를 넣어야 합니다.
- 즉 학습은 JSONL(토큰 시퀀스)로 하고, 추론은 SVG 원본으로 실행합니다.

### test.jsonl 샘플 1개를 실제 추론에 쓰고 싶을 때
1. JSONL row를 outline SVG로 복원
```bash
python inference/generate_from_jsonl.py --jsonl dataset/processed/test.jsonl --sample_id 46 --output_svg tmp_outline_46.svg
```
2. 복원한 SVG를 모델에 입력
```bash
python inference/generate.py --input_svg tmp_outline_46.svg --adapter_dir checkpoints/byt5_lora --tokenizer_dir checkpoints/byt5_lora --output_svg pred_46.svg
```

### size mismatch 오류 원인과 해결
- 원인: 학습 시 `resize_token_embeddings(len(custom_tokenizer))`를 했는데, 추론 시 base model만 로드하면 vocab 크기가 달라져 adapter 로드가 실패.
- 해결: 추론에서도 동일 tokenizer 로드 후 `resize_token_embeddings(len(tok))`를 먼저 수행한 뒤 LoRA adapter를 로드해야 함.
- 본 저장소의 `inference/generate.py`는 이 순서를 반영했고, 학습 시 `train_meta.json`을 저장해 base model 추적 가능.
