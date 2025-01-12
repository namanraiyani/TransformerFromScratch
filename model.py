import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model) #Postional Encoding
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # Vector shaped (seq_len, 1)
        division_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position*division_term) # Apply sin to even positions
        pe[:, 1::2] = torch.cos(position*division_term) # Apply cos to odd positions
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires._grad_(False)
        return self.dropout(x)