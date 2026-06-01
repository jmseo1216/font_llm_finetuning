# from __future__ import annotations

# import json
# from pathlib import Path
# from torch.utils.data import Dataset


# class SVGSeq2SeqDataset(Dataset):
#     def __init__(self, jsonl_path: str):
#         self.rows = [json.loads(x) for x in Path(jsonl_path).read_text(encoding="utf-8").splitlines()]

#     def __len__(self):
#         return len(self.rows)

#     def __getitem__(self, idx):
#         r = self.rows[idx]
#         src = "<OUTLINE> " + " ".join(r["outline_tokens"])
#         tgt = "<SKELETON> " + " ".join(r["skeleton_tokens"])
#         return {"id": r["id"], "src_text": src, "tgt_text": tgt, "width": r["width"], "height": r["height"]}




## quantize_coordinates 사용하지 않는 버전 
from __future__ import annotations

import json
from pathlib import Path
from torch.utils.data import Dataset


class SVGSeq2SeqDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows = []

        path = Path(jsonl_path)

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]

        # =====================================================
        # 1. raw SVG text 방식
        # build_dataset_raw.py 결과:
        # {
        #   "src_text": "outline2skeleton: <OUTLINE> M ...",
        #   "tgt_text": "<SKELETON> M ..."
        # }
        # =====================================================
        if "src_text" in r and "tgt_text" in r:
            src = r["src_text"]
            tgt = r["tgt_text"]

        # =====================================================
        # 2. 기존 quantized token 방식
        # build_dataset.py 결과:
        # {
        #   "outline_tokens": ["<CMD_M>", "<NUM_...>", ...],
        #   "skeleton_tokens": [...]
        # }
        # =====================================================
        elif "outline_tokens" in r and "skeleton_tokens" in r:
            src = "<OUTLINE> " + " ".join(r["outline_tokens"])
            tgt = "<SKELETON> " + " ".join(r["skeleton_tokens"])

        else:
            raise KeyError(
                "JSONL row must contain either "
                "('src_text', 'tgt_text') or "
                "('outline_tokens', 'skeleton_tokens'). "
                f"Available keys: {list(r.keys())}"
            )

        return {
            "id": r.get("id", str(idx)),
            "src_text": src,
            "tgt_text": tgt,
            "width": r.get("width", 50.0),
            "height": r.get("height", 50.0),
        }