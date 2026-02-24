import torch
import torch.nn as nn
from .base_rnn import BaseRNN

class EncoderRNN(BaseRNN):
    """
    EncoderRNN
    Inputs:

    Outputs: [B,S,H*dir] if batch_first=True else [S,B,H*dir]
    Hidden: final hidden state (or (h,c) for LSTM)
    """
    def __init__(self, vocab_size, hidden_size, n_layers=1, 
            dropout_p=0.1, cell_type='GRU', 
            batch_first=False, padding_idx=None, 
            embedding_size=None, bidirectional=False):
        
        super(EncoderRNN, self).__init__(
            vocab_size=vocab_size, hidden_size=hidden_size, n_layers=n_layers, 
            dropout_p=dropout_p, cell_type=cell_type, 
            batch_first=batch_first, padding_idx=padding_idx, 
            embedding_size=embedding_size, bidirectional=bidirectional
        )
    
    def forward(self, input_seq, input_lengths=None, hidden=None):
        """
        forward 的 Docstring
        parameters
            input_seq: 
                - batch_first=False: [S, B]
                - batch_first=True : [B, S]
            input_lengths: [batch_size]
            hidden: 初始隐藏状态
        return:
            outputs: [seq_len, batch, hidden_size * num_directions]
            hidden: [n_layers * num_directions, batch, hidden_size]
        """
        if hidden is None:
            batch_size = input_seq.size(0) if self.batch_first else input_seq.size(1)
            hidden = self.init_hidden(batch_size)

        # Embedding
        # shape: [S,B,E] or [B,S,E]
        embedded = self.embedding(input_seq) 

        # Dropout
        embedded = self.dropout_layer(embedded)

        # Packed Sequence
        if input_lengths is not None:
            # 注意：输入序列必须按长度降序排列才能使用 pack_padded_sequence
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, input_lengths.cpu(), batch_first=self.batch_first,
                enforce_sorted=False
            )

        # RNN forward
        outputs, hidden = self.rnn(embedded, hidden)

        # Unpack
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=self.batch_first
            )
        
        return outputs, hidden

