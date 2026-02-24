import unicodedata
import re
import jieba

def unicode_to_ascii(s: str) ->str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

def normalize_string(s: str) ->str:
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z\u4e00-\u9fa5.!?，。？]+", r" ", s)
    s = re.sub(r"\s+", " ", s).strip() # 匹配连续的空白字符并替换为1个空格
    return s

def tokenize_cn(s, level="char"):
    s = s.strip()
    if level == "char":
        return list(s.replace(" ", ""))
    elif level == "word":
        tokens = list(jieba.cut(s, cut_all=False))
        tokens = [t for t in tokens if t.strip()]
        return tokens
    else:
        raise ValueError(f"Unknown cn level: {level}")

def tokenize_en(s: str):
    return s.strip().split()
