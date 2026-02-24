import torch
import torch.nn as nn

class BaseRNN(nn.Module):
    """
    A shared base for RNN/GRU/LSTM encoder/decoder.
    """
    def __init__(
        self,
        vocab_size, hidden_size, n_layers=1, dropout_p=0.1, cell_type='GRU',
        batch_first=False, padding_idx=None, embedding_size=None, bidirectional=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size if embedding_size is not None else hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.cell_type = cell_type.upper()
        self.batch_first = batch_first
        self.padding_idx = padding_idx   
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        # RNN Layer
        allowed_cells = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}
        if self.cell_type not in allowed_cells:
            raise ValueError(f"Cell type must be one of {list(allowed_cells.keys())}")
        
        # Embedding Layer
        self.embedding = nn.Embedding(
            vocab_size,
            self.embedding_size,
            padding_idx=self.padding_idx
        )

        # Dropout Layer
        self.dropout_layer = nn.Dropout(self.dropout_p)


        rnn_cls = allowed_cells[self.cell_type]

        self.rnn = rnn_cls(
            input_size = self.embedding_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=self.dropout_p if n_layers > 1 else 0.0,
            batch_first=self.batch_first,
            bidirectional=self.bidirectional,
        )
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("BaseRNN is an abstract class, do not use directly.")
    
    def init_hidden(self, batch_size):
        """
        初始化隐藏状态
        RNN/GRU: h0 shape (n_layers, batch_size, hidden_size)
        LSTM: (h0, c0)
        """
        # 获取模型第一个参数的引用
        param = next(self.parameters())
        h0 = param.new_zeros(self.n_layers * self.num_directions, batch_size, self.hidden_size)

        if self.cell_type == "LSTM":
            c0 = param.new_zeros(self.n_layers * self.num_directions, batch_size, self.hidden_size)
            return (h0, c0)
        
        return h0



        