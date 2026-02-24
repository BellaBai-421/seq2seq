import torch
import torch.nn as nn
import random

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, sos_id=0, eos_id=1, pad_id=2):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        assert encoder.batch_first == decoder.batch_first, "encoder/decoder batch_first must match"
        assert not decoder.bidirectional, "decoder must be unidirectional"

        # hidden_size 相同; bidirectional encoder 需要birdge
        assert encoder.hidden_size == decoder.hidden_size, "hidden_size mismatch"

        # 确保 cell_type 一致
        assert encoder.cell_type == decoder.cell_type, "cell_type mismatch"
    
    def _bridge_bidirectional_hidden(self, hidden):
        if not self.encoder.bidirectional:
            return hidden
        
        def merge(h):
            f = h[0::2]  # [L, B, H]
            b = h[1::2]  # [L, B, H]
            return f + b # 是否改成concat+linear会更好？
        
        if self.encoder.cell_type == "LSTM":
            h, c = hidden
            return (merge(h), merge(c))
        return merge(hidden)
    
    def forward(self, src, trg, src_lengths=None, teacher_forcing_ratio=0.5):
        """
        forward 的 Docstring
        
        :param src: [S, B] or [B, S]
        :param trg: [T, B] or [B, T]
        :param src_lengths: 包含源序列真实长度的 1D Tensor，用于 pack_padded_sequence
        """
        # src: [S, B] (batch_first=False)
        # trg: [T, B]
        device = src.device

        batch_size = src.size(0) if self.encoder.batch_first else src.size(1)
        trg_len = trg.size(1) if self.decoder.batch_first else trg.size(0)
        vocab_size = self.decoder.vocab_size

        all_logits = []

        _, enc_hidden = self.encoder(src, input_lengths=src_lengths)

        hidden = self._bridge_bidirectional_hidden(enc_hidden)

        # decoder input: 获取 trg 的第一个时间步 (应为 <SOS>)
        if self.decoder.batch_first:
            decoder_input = trg[:, 0]  # [B]
        else:
            decoder_input = trg[0, :]  # [B]

        for t in range(1, trg_len):
            log_probs, hidden = self.decoder.step(decoder_input, hidden)

            all_logits.append(log_probs.unsqueeze(1 if self.decoder.batch_first else 0))

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = log_probs.argmax(dim=1) # [B]

            if teacher_force:
                decoder_input = trg[:, t] if self.decoder.batch_first else trg[t, :]
            
            else:
                decoder_input = top1
        
        # 结果形状为 [B, T-1, V] 或 [T-1, B, V]
        return torch.cat(all_logits, dim=1 if self.decoder.batch_first else 0)
    
    @torch.no_grad()
    def translate_greedy(self, src, src_lengths, max_len=25):
        self.eval()
        batch_size = src.size(0) if self.encoder.batch_first else src.size(1)
        _, enc_hidden = self.encoder(src, src_lengths)
        hidden = self._bridge_bidirectional_hidden(enc_hidden)

        # 初始输入为 <sos>
        decoder_input = torch.full((batch_size,), self.sos_id, device=src.device)
        decoded_indices = []

        for _ in range(max_len):
            log_probs, hidden = self.decoder.step(decoder_input, hidden)
            top1 = log_probs.argmax(1) 
            decoded_indices.append(top1.unsqueeze(1))
            decoder_input = top1

        # [B, max_len]
        return torch.cat(decoded_indices, dim=1)
    
    @torch.no_grad()
    def translate_beam_search(self, src, src_lengths, beam_size=3, max_len=25):
        self.eval()
        _, enc_hidden = self.encoder(src, src_lengths)
        hidden = self._bridge_bidirectional_hidden(enc_hidden)

        # 累积对数概率, 已生成的词序列, 隐藏状态
        beams = [(0.0, [self.sos_id], hidden)]

        for _ in range(max_len):
            candidates = []
            for score, seq, h in beams:
                if seq[-1] == self.eos_id:
                    candidates.append((score, seq, h))
                    continue

                inp = torch.tensor([seq[-1]], device=src.device)
                log_probs, next_h = self.decoder.step(inp, h)

                # 取当前路径下概率最大的 K 个词
                topk_v, topk_i = log_probs.topk(beam_size)
                for i in range(beam_size):
                    candidates.append((score + topk_v[0][i].item(), seq + [topk_i[0][i].item()], next_h))
            
            # 全局排序并保留前 beam_size 个
            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

            if all(b[1][-1] == self.eos_id for b in beams):
                break
        
        return beams[0][1] # 返回得分最高的一条 [List of IDs]
    

