# BLEU
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch

def evaluate_bleu(model, dataloader, vocab_tgt, device, max_len=25):
    model.eval()
    refs = []
    cands = []
    sf = SmoothingFunction().method1

    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            src_len = batch["src_lengths"].to("cpu")   # 最稳
            tgt = batch["tgt"]                         # 留在CPU即可

            pred_ids = model.translate_greedy(src, src_len, max_len=max_len)  # [B, Tpred]

            B = pred_ids.size(0)
            for i in range(B):
                cand = vocab_tgt.decode(pred_ids[i].tolist(), stop_at_eos=True)

                # reference: 丢<sos>，在<eos>截断
                ref = vocab_tgt.decode(tgt[i].tolist(), stop_at_eos=True)
                if len(ref) > 0 and ref[0] == vocab_tgt.specials.SOS:
                    ref = ref[1:]

                cands.append(cand)
                refs.append([ref])

    bleu4 = corpus_bleu(refs, cands, smoothing_function=sf) * 100
    return bleu4
