# padding成相同长度张量
# 在CPU上计算，可能有运算速度问题
from typing import List, Dict
import torch

def _pad(seqs: List[List[int]], pad_id: int):
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lens.max().item()) if len(seqs) else 0
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out, lens

def make_collate_fn(src_pad_id: int, tgt_pad_id: int, batch_first: bool):
    def collate(batch: List[Dict]):
        src_seqs = [b["src_ids"] for b in batch]
        tgt_seqs = [b["tgt_ids"] for b in batch]

        src, src_lengths = _pad(src_seqs, src_pad_id) # [B, S]
        tgt, tgt_lengths = _pad(tgt_seqs, tgt_pad_id) # [B, T]

        if not batch_first:
            src = src.transpose(0, 1).contiguous() # [S, B]
            tgt = tgt.transpose(0, 1).contiguous() # [T, B]
        
        return {"src": src, "src_lengths": src_lengths, "tgt": tgt, "tgt_lengths": tgt_lengths}
    return collate

