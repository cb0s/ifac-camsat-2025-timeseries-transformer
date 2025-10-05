import math

import torch
from torch import nn


class InputEmbedding(nn.Module):
    """
    Projects the input features into the model's dimension (d_model).
    """
    def __init__(self, num_features, d_model):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(num_features, d_model)

    def forward(self, x):
        # The input x has shape (batch_size, seq_len, num_features)
        # We project it to (batch_size, seq_len, d_model)
        return self.linear(x) * math.sqrt(self.d_model)


class PositionalEncodingDecoderOnly(nn.Module):
    """ Standard Sinusoidal Positional Encoding """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model, requires_grad=False)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Transpose to (1, max_len, d_model) for batch_first=True compatibility
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)  # Register as buffer so it moves with the model to device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # Add positional encoding up to the length of the sequence
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding.
    This is added to the input embeddings to provide the model with position information.
    """

    def __init__(self, d_model, dropout_rate=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Create a positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        # Add a batch dimension: [1, max_len, d_model]
        # This allows for easy broadcasting to the input tensor's batch size.
        pe = pe.unsqueeze(0)

        # Register 'pe' as a buffer. Buffers are part of the model's state,
        # but they are not considered model parameters (i.e., not trained).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input embedding. Shape: (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: The input embedding + positional encoding.
        """
        # Add positional encoding to the input tensor x.
        # self.pe is [1, max_len, d_model]. We slice it to match the input sequence length.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
