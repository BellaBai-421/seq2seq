# translate.py
import argparse
from pathlib import Path
import torch

from dataio.vocab import Vocab
from utils.text import normalize_string, tokenize_cn, tokenize_en
from utils.checkpoint import load_checkpoint

from models.encoder_rnn import EncoderRNN
from models.decoder_rnn import DecoderRNN
from models.seq2seq import Seq2seq


@torch.no_grad()
def greedy_decode(model: Seq2seq, src, src_lengths, sos_id: int, eos_id: int, max_len: int, batch_first: bool):
    model.eval()
    device = src.device

    _, enc_hidden = model.encoder(src, input_lengths=src_lengths)
    hidden = model._bridge_bidirectional_hidden(enc_hidden)  # 你 seq2seq 里已有这个方法

    batch_size = src.size(0) if batch_first else src.size(1)
    y = torch.full((batch_size,), sos_id, dtype=torch.long, device=device)

    outputs = []
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        log_probs, hidden = model.decoder.step(y, hidden)  # [B,V]
        y = log_probs.argmax(dim=-1)                      # [B]
        outputs.append(y)
        finished |= (y == eos_id)
        if finished.all():
            break

    # [B, T]
    return torch.stack(outputs, dim=1) if outputs else torch.empty((batch_size, 0), dtype=torch.long, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="outputs/run1/checkpoint.pt")
    ap.add_argument("--vocab", type=str, default="outputs/run1/vocabs.pt")
    ap.add_argument("--test", type=str, default="data/test.txt")
    ap.add_argument("--out", type=str, default="outputs/run1/test_pred.txt")
    ap.add_argument("--cn_level", type=str, default="char", choices=["char", "word"])
    ap.add_argument("--max_len", type=int, default=50)
    ap.add_argument("--batch_first", action="store_true")
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--cell", type=str, default="GRU", choices=["RNN", "GRU", "LSTM"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocabs = torch.load(args.vocab, map_location="cpu")
    src_vocab = Vocab()
    tgt_vocab = Vocab()
    # 复原 itos/stoi（固定 id 的前提下可直接覆盖）
    src_vocab.itos = vocabs["src_itos"]
    src_vocab.stoi = {t:i for i,t in enumerate(src_vocab.itos)}
    tgt_vocab.itos = vocabs["tgt_itos"]
    tgt_vocab.stoi = {t:i for i,t in enumerate(tgt_vocab.itos)}
    src_vocab.sos_id = src_vocab.stoi["<sos>"]
    src_vocab.eos_id = src_vocab.stoi["<eos>"]
    src_vocab.pad_id = src_vocab.stoi["<pad>"]
    src_vocab.unk_id = src_vocab.stoi["<unk>"]
    tgt_vocab.sos_id = tgt_vocab.stoi["<sos>"]
    tgt_vocab.eos_id = tgt_vocab.stoi["<eos>"]
    tgt_vocab.pad_id = tgt_vocab.stoi["<pad>"]
    tgt_vocab.unk_id = tgt_vocab.stoi["<unk>"]

    encoder = EncoderRNN(
        vocab_size=len(src_vocab),
        hidden_size=args.hidden,
        n_layers=args.layers,
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
    model = Seq2seq(encoder, decoder, sos_id=tgt_vocab.sos_id, eos_id=tgt_vocab.eos_id).to(device)

    load_checkpoint(args.ckpt, model, optimizer=None, map_location=device)

    # 逐行翻译（作业 test.txt 每行一句中文）
    out_lines = []
    with open(args.test, "r", encoding="utf-8") as f:
        for line in f:
            cn = normalize_string(line.strip())
            if not cn:
                out_lines.append("")
                continue

            src_toks = tokenize_cn(cn, level=args.cn_level)
            src_ids = [src_vocab.stoi.get(t, src_vocab.unk_id) for t in src_toks] + [src_vocab.eos_id]

            src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device)
            if args.batch_first:
                src_tensor = src_tensor.unsqueeze(0)   # [B=1,S]
            else:
                src_tensor = src_tensor.unsqueeze(1)   # [S,B=1]

            src_lengths = torch.tensor([len(src_ids)], dtype=torch.long, device=device)

            pred_ids = greedy_decode(
                model, src_tensor, src_lengths,
                sos_id=tgt_vocab.sos_id, eos_id=tgt_vocab.eos_id,
                max_len=args.max_len, batch_first=args.batch_first
            )[0].tolist()

            pred_tokens = tgt_vocab.decode(pred_ids, stop_at_eos=True)
            # 英文按空格拼回去
            out_lines.append(" ".join(pred_tokens))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for l in out_lines:
            f.write(l + "\n")

    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()

# python translate.py \
#   --ckpt outputs/run1/checkpoint.pt \
#   --vocab outputs/run1/vocabs.pt \
#   --test data/test.txt \
#   --out outputs/run1/test_pred.txt \
#   --bidirectional --cell GRU --hidden 256 --layers 1
