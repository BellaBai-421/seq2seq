import torch
import torch.nn as nn
import random

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, sos_id=0, eos_id=1):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_id = sos_id
        self.eos_id = eos_id

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

        # [T, B, V]
        outputs = torch.zeros(trg_len, batch_size, vocab_size, device=device)

        _, enc_hidden = self.encoder(src, input_lengths=src_lengths)

        hidden = self._bridge_bidirectional_hidden(enc_hidden)

        # decoder input: 获取 trg 的第一个时间步 (应为 <SOS>)
        if self.decoder.batch_first:
            decoder_input = trg[:, 0]  # [B]
        else:
            decoder_input = trg[0, :]  # [B]

        # # decoder input: <sos> or trg[0]
        # decoder_input = trg[0] # [B](这里要求 trg 已经以 <sos> 开头)

        for t in range(1, trg_len):
            log_probs, hidden = self.decoder.step(decoder_input, hidden)
            outputs[t] = log_probs

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = log_probs.argmax(dim=1) # [B]

            if teacher_force:
                decoder_input = trg[:, t] if self.decoder.batch_first else trg[t, :]
            
            else:
                decoder_input = top1
        
        if self.decoder.batch_first:
            outputs = outputs.transpose(0, 1)
        
        return outputs

