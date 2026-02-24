# train
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataio.dataset import read_cn_en_pairs, filter_pairs, build_vocabs, CNENDataset, split_train_dev
from dataio.collate import make_collate_fn
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint

from models.encoder_rnn import EncoderRNN
from models.decoder_rnn import DecoderRNN
from models.seq2seq import Seq2seq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="data/cn-eng.txt")
    ap.add_argument("--out_dir", type=str, default="outputs/run1") # test name
    ap.add_argument("--max_len", type=int, default=10)
    ap.add_argument("--dev_ratio", type=float, default=0.05) # dev ratio
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cn_level", type=str, default="char", choices=["char", "word"])

    ap.add_argument("--cell", type=str, default="RNN", choices=["RNN", "GRU", "LSTM"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--batch_first", action="store_true")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10) # 原本训练中是多少？
    ap.add_argument("--teacher_forcing", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=5.0)

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读数据 + 过滤
    pairs = read_cn_en_pairs(args.data_path)
    pairs = filter_pairs(pairs, max_len=args.max_len, cn_level=args.cn_level)

    train_pairs, dev_pairs = split_train_dev(pairs, dev_ratio=args.dev_ratio, seed=args.seed)

    # 建词表
    src_vocab, tgt_vocab = build_vocabs(train_pairs, cn_level=args.cn_level)

    # save vocab
    torch.save({"src_itos": src_vocab.itos, "tgt_itos": tgt_vocab.itos}, out_dir / "vocabs.pt")

    # load data
    train_ds = CNENDataset(train_pairs, src_vocab, tgt_vocab, cn_level=args.cn_level, max_len=None) # ?为什么不是参数中的max_len
    dev_ds = CNENDataset(dev_pairs, src_vocab, tgt_vocab, cn_level=args.cn_level, max_len=None)

    collate = make_collate_fn(src_vocab.pad_id, tgt_vocab.pad_id, batch_first=args.batch_first)
    train_loader = DataLoader(train_ds, batch_first=args.batch_first, shuffle=True, collate_fn=collate) # ?这里是什么？为什么可以这么写
    dev_loader = DataLoader(dev_ds, batch_first=args.batch_first, shuffle=False, collate_fn=collate)

    # 构建模型
    encoder = EncoderRNN(
        vocab_size=len(src_vocab),
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout_p=args.dropout,
        cell_type=args.cell,
        batch_first=args.batch_first,
        padding_idx=src_vocab.pad_id,
        embedding_size=None,
        bidirectional=args.bidirectional,
    )
    decoder = DecoderRNN(
        vocab_size=len(tgt_vocab),
        hidden_size=args.hidden,
        n_layers=args.layers,
        dropout_p=args.dropout,
        cell_type=args.cell,
        batch_first=args.batch_first,
        padding_idx=tgt_vocab.pad_id,
        embedding_size=None,
        bidirectional=False,
    )
    model = Seq2seq(encoder, decoder, sos_id=tgt_vocab.sos_id, eos_id=tgt_vocab.eos_id)

    model.to(device)

    criterion = nn.NLLLoss(ignore_index=tgt_vocab.pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def run_eval():
        model.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for batch in dev_loader:
                src = batch["src"].to(device)
                trg = batch["trg"].to(device)
                src_lengths = batch["src_lengths"].to(device)

                outputs = model(src, trg, src_lengths=src_lengths, teacher_forcing_ratio=0.0)

                if args.batch_first:
                    # outputs [B,T,V], trg [B,T]
                    out = outputs[:, 1:, :].contiguous().view(-1, outputs.size(-1))
                    gold = trg[:, 1:].contiguous().view(-1)
                else:
                    # outputs [T,B,V], trg [T,B]
                    out = outputs[1:, :, :].contiguous().view(-1, outputs.size(-1))
                    gold = trg[1:, :].contiguous().view(-1)
                
                loss = criterion(out, gold)
                # 统计有效 token
                n_tokens = int((gold != tgt_vocab.pad_id).sum().item())
                total_loss += loss.item() * max(n_tokens, 1)
                total_tokens += n_tokens
        
        model.train()
        return total_loss / max(total_tokens, 1) # 这里的return是什么？为什么这么计算？
    
    # train loop
    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch in train_loader:
            src = batch["src"].to(device)
            trg = batch["trg"].to(device) # 这里为什么使用trg？这里的batch是什么函数的？为什么不使用tgt？
            src_lengths = batch["src_lengths"].to(device)

            outputs = model(src, trg, src_lengths=src_lengths, teacher_forcing_ratio=args.teacher_forcing)

            if args.batch_first:
                out = outputs[:, 1:, :].contiguous().view(-1, outputs.size(-1))
                gold = trg[:, 1:].contiguous().view(-1)
            else:
                out = outputs[1:, :, :].contiguous().view(-1, outputs.size(-1))
                gold = trg[1:, :].contiguous().view(-1)
            
            loss = criterion(out, gold)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) # 为什么需要clip？
            optimizer.step()

            global_step += 1
            
            if global_step % 200 == 0:
                dev_loss = run_eval()
                print(f"epoch={epoch} step={global_step} train_loss={loss.item():.4f} dev_loss={dev_loss:.4f}") # 需要增加计时器

        save_checkpoint(
            run_dir=str(out_dir),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
            extra={
                "src_vocab_size": len(src_vocab),
                "tgt_vocab_size": len(tgt_vocab),
                "pad_id": tgt_vocab.pad_id,
                "sos_id": tgt_vocab.sos_id,
                "eos_id": tgt_vocab.eos_id,
                "args": vars(args),
            },
        )
        print(f"[saved] epoch {epoch} to {out_dir / 'checkpoint.pt'}")

if __name__ == "__main__":
    main()

# python train.py \
#   --data_path data/cn-eng.txt \
#   --out_dir outputs/run1 \
#   --cell GRU --hidden 256 --layers 1 --dropout 0.1 \
#   --bidirectional \
#   --batch_size 64 --lr 1e-3 --epochs 10 --teacher_forcing 0.5






