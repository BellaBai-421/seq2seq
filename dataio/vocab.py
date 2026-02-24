from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class Specials:
    SOS: str = "<sos>"
    EOS: str = "<eos>"
    PAD: str = "<pad>"
    UNK: str = "<unk>"

class Vocab:
    """
    固定 special token 的 id, 避免训练/推理不一致。
    """
    def __init__(self, specials: Specials = Specials()): 
        self.specials = specials
        self.itos: List[str] = [] # itos: index to string
        self.stoi: Dict[str, int] = {} # stoi: string to index

        for tok in [specials.SOS, specials.EOS, specials.PAD, specials.UNK]:
            self.add(tok)
        
        self.sos_id = self.stoi[specials.SOS]
        self.eos_id = self.stoi[specials.EOS]
        self.pad_id = self.stoi[specials.PAD]
        self.unk_id = self.stoi[specials.UNK]
    
    def __len__(self):
        return len(self.itos)
    
    def add(self, tok: str) -> int:
        if tok not in self.stoi:
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)
        return self.stoi[tok]
    
    def add_many(self, toks: List[str]):
        for t in toks:
            self.add(t)
    
    def encode(self, toks: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_id) for t in toks]
    
    def decode(self, ids: List[int], stop_at_eos: bool = True) -> List[str]:
        out = []
        for i in ids:
            tok = self.itos[i] if 0 <= i < len(self.itos) else self.specials.UNK
            if stop_at_eos and tok == self.specials.EOS:
                break
            out.append(tok)
        return out
    
    def state_dict(self) -> dict:
        return {"itos": self.itos}
    
    @ classmethod
    def from_state_dict(cls, sd: dict):
        v = cls()
        v.itos = sd["itos"]
        v.stoi = {t:i for i,t in enumerate(v.itos)}
        v.sos_id = v.stoi[v.specials.SOS]
        v.eos_id = v.stoi[v.specials.EOS]
        v.pad_id = v.stoi[v.specials.PAD]
        v.unk_id = v.stoi[v.specials.UNK]
        return v
