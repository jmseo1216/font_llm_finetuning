from __future__ import annotations

import json
from pathlib import Path
from torch.utils.data import Dataset


class SVGSeq2SeqDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.rows = [json.loads(x) for x in Path(jsonl_path).read_text(encoding="utf-8").splitlines()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        src = "<OUTLINE> " + " ".join(r["outline_tokens"])
        tgt = "<SKELETON> " + " ".join(r["skeleton_tokens"])
        return {"id": r["id"], "src_text": src, "tgt_text": tgt, "width": r["width"], "height": r["height"]}
