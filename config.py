# hyperparameters
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import time
import torch

@dataclass
class Config:
    # paths
    data_dir: str = "data"
    train_file: str = "cn-eng.txt"
    test_file: str = "test.txt"
    out_dir: str = "outputs"
    exp_name: str = "rnn_baseline"

    # data
    max_len: int = 10
    dev_ratio: float = 0.05
    seed: int = 42
    cn_level: str = "char"   # "char" or "word"(你后面可扩展分词)

    input_vocab_size: int = 0
    output_vocab_size: int = 0

    # model
    model_type: str = "rnn"  # rnn/gru/lstm
    hidden_size: int = 256
    n_layers: int = 2
    dropout: float = 0.1

    bidirectional: bool = False

    # train
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 20
    teacher_forcing: float = 0.5
    clip: float = 5.0

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # decode
    max_decode_len: int = 20
    beam_width: int = 1

    # tensorboard
    use_tensorboard: bool = True
    tb_subdir: str = "tb"           # under run_dir, e.g. run_dir/tb
    log_every: int = 50             # steps

    def run_dir(self) -> Path:
        ts = time.strftime("%Y%m%d-%H%M%S")
        return Path(self.out_dir) / "runs" / f"{self.exp_name}-{ts}"
    
    def tb_dir(self, run_dir: Path) -> Path:
        return run_dir / self.tb_subdir

    def save(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
