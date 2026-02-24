import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_rnn import BaseRNN

class DecoderRNN(BaseRNN):
    """
    DecoderRNN 的 Docstring
    """
    def __init__(self, vocab_size, hidden_size, n_layers=1, dropout_p=0.1, cell_type='GRU', batch_first=False, padding_idx=None, embedding_size=None, bidirectional=False):
        super().__init__(vocab_size=vocab_size, hidden_size=hidden_size, 
                         n_layers=n_layers, dropout_p=dropout_p, 
                         cell_type=cell_type, batch_first=batch_first, 
                         padding_idx=padding_idx, embedding_size=embedding_size, bidirectional=False)
        
        # Output projection layer (from hidden_size -> vocab_size)
        self.out_proj = nn.Linear(self.hidden_size, self.vocab_size)
    
    def step(self, y_prev, hidden):
        """
        parameters:
        :param y_prev: [B] or [B, 1] token ids (previous token input to the decoder)
        :type y_prev: torch.Tensor
        :param hidden: [n_layers * num_directions, B, H] hidden state (or (h,c) for LSTM)
        :type hidden: torch.Tensor

        returns:
        log_probs: [B, V] log probabilities of next token (V = vocab_size)
        hidden: updated hidden state
        """
        # Ensure y_prev is shaped correctly: [B] -> [1,B] or [B,1] as expected by embedding
        if y_prev.dim() == 1: # [B]
            if self.batch_first:
                y_prev = y_prev.unsqueeze(1) # [B, 1]
            else:
                y_prev = y_prev.unsqueeze(0) # [1, B]
        
        elif y_prev.dim() == 2: # [B,1] or [1,B]
            pass
        else:
            raise ValueError(f"y_prev must be of shape [B] or [B,1], got {y_prev.shape}")
        
        # Embedding lookup
        emb = self.embedding(y_prev) # [1,B,E] or [B,1,E]
        emb =self.dropout_layer(emb)

        # Pass through RNN
        out, hidden = self.rnn(emb, hidden) # out: [1,B,H] or [B,1,H]

        # Take the output at the only time step(because we inputted a single token)
        out_step = out.squeeze(1) if self.batch_first else out.squeeze(0) # [B, H]

        # Projection hidden to vocab size
        logits = self.out_proj(out_step) # [B, V]
        log_probs = F.log_softmax(logits, dim=-1) # [B, V]

        return log_probs, hidden

 
