# 读数据、建词表
from typing import List, Tuple, Optional
import random

from torch.utils.data import Dataset
from utils.text import normalize_string, tokenize_cn, tokenize_en
from dataio.vocab import Vocab

Pair = Tuple[str, str]

def read_cn_en_pairs(path: str) -> List[Pair]:
    pairs: List[Pair] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 原作业：cn \t en
            cn, en = line.split("\t")
            cn = normalize_string(cn)
            en = normalize_string(en)
            pairs.append((cn, en))
    return pairs

def filter_pairs(pairs: List[Pair], max_len: int = 10, cn_level: str = "char") -> List[Pair]:
    out = []
    for cn, en in pairs:
        cn_len = len(tokenize_cn(cn, level=cn_level))
        en_len = len(tokenize_en(en))
        if cn_len < max_len and en_len < max_len:
            out.append((cn, en))
    return out

def build_vocabs(
        pairs: List[Pair],
        cn_level: str = "char"
) -> Tuple[Vocab, Vocab]:
    src_vocab = Vocab()
    tgt_vocab = Vocab()

    for cn, en in pairs:
        src_vocab.add_many(tokenize_cn(cn, level=cn_level))
        tgt_vocab.add_many(tokenize_en(en))
        
    return src_vocab, tgt_vocab

class CNENDataset(Dataset):
    """
    返回数值化后的 src_ids / tgt_ids
    - src: tokens + <eos>
    - tgt: <sos> + tokens + <eos>
    """
    def __init__(
            self, pairs: List[Pair], 
            src_vocab: Vocab, tgt_vocab: Vocab, 
            cn_level: str="char",
            max_len: Optional[int] = None):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.cn_level = cn_level
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx: int):
        cn, en = self.pairs[idx]
        src_tok = tokenize_cn(cn, level=self.cn_level)
        tgt_tok = tokenize_en(en)

        if self.max_len is not None:
            src_tok = src_tok[: self.max_len]
            tgt_tok = tgt_tok[: self.max_len]

        src_ids = self.src_vocab.encode(src_tok) + [self.src_vocab.eos_id]
        tgt_ids = [self.tgt_vocab.sos_id] + self.tgt_vocab.encode(tgt_tok) + [self.tgt_vocab.eos_id]

        return {"src_ids": src_ids, "tgt_ids": tgt_ids}

def split_train_dev(pairs: List[Pair], dev_ratio: float, seed: int = 42):
    rnd = random.Random(seed)
    pairs = pairs[:]
    rnd.shuffle(pairs)
    n_dev = int(len(pairs) * dev_ratio)
    dev_pairs = pairs[:n_dev]
    train_pairs = pairs[n_dev:]

    return train_pairs, dev_pairs


