# 当前实现的checkpoint功能是否足够？
from pathlib import Path
import torch

def save_checkpoint(run_dir: str, model, optimizer, epoch: int, step: int, extra: dict):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "extra": extra, # ?这个是什么
    }
    torch.save(ckpt, run_dir / "checkpoint.pt")

def load_checkpoint(path: str, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt




