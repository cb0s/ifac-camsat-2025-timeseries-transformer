import math

import torch
from torch import nn
import torch.nn.functional as F

from src.model.encoder import PositionalEncodingDecoderOnly, InputEmbedding, PositionalEncoding


class TimeSeriesDecoderOnlyTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_seq_len: int,
                 d_model: int,
                 n_heads: int,
                 num_decoder_layers: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 max_output_len: int = 500,  # Max length for positional encoding
                 device='cuda'):
        super().__init__()
        self.input_dim = input_dim
        self.output_seq_len = output_seq_len
        self.d_model = d_model

        # 1. Input Projection: Project the 904-dim input vector to d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # 2. Target Embedding: Project the 1D output step back to d_model
        # This is needed for the auto-regressive input during generation/training
        self.target_embedder = nn.Linear(1, d_model)  # Assuming 1 feature per output step

        # 3. Positional Encoding
        self.pos_encoder = PositionalEncodingDecoderOnly(d_model, dropout, max_len=max_output_len)

        # 4. Transformer Decoder Layers
        # We use the standard DecoderLayer but will feed the input context
        # into the 'memory' argument during the forward pass.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',  # GELU often preferred in modern transformers
            batch_first=True,  # Ensure batch dimension comes first
            norm_first=True  # Pre-LN potentially offers better stability
        )

        # 5. Transformer Decoder Stack
        # LayerNorm is applied *after* the stack if specified
        decoder_norm = nn.LayerNorm(d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=decoder_norm
        )

        # 6. Output Layer: Project decoder output back to 1 dimension per timestep
        self.output_projection = nn.Linear(d_model, 1)  # Output 1 residual value per step

        self._init_weights()
        self._device = torch.device(device)

    def _init_weights(self):
        # Initialize weights for better convergence
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # --- Time Series Decoder-Only Transformer ---
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.empty(sz, sz, device=self._device).fill_(float('-inf')), diagonal=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Training pass using Teacher Forcing.

        Args:
            src: Input tensor, shape [batch_size, input_dim (904)]
            tgt: Target tensor, shape [batch_size, output_seq_len (432), 1]
                   Represents the ground truth sequence.

        Returns:
            Output tensor, shape [batch_size, output_seq_len, 1]
        """
        # 1. Project input context vector
        # src: [B, 904] -> projected_context: [B, D_MODEL]
        projected_context = self.input_projection(src)

        # 2. Prepare decoder input (teacher forcing)
        # We need input for steps 0 to L-1 to predict steps 1 to L.
        # Prepend the projected context as the representation for step 0.
        # tgt: [B, L, 1]
        # Use tgt[:, :-1] for input embedding
        tgt_embedded = self.target_embedder(tgt[:, :-1, :])  # [B, L-1, D_MODEL]

        # Combine context (as step 0 input) and embedded targets
        # projected_context.unsqueeze(1): [B, 1, D_MODEL]
        # decoder_input_embedded: [B, 1 + (L-1), D_MODEL] = [B, L, D_MODEL]
        decoder_input_embedded = torch.cat([projected_context.unsqueeze(1), tgt_embedded], dim=1)  # [B, L, D_MODEL]

        # 3. Add positional encoding
        decoder_input_pos = self.pos_encoder(decoder_input_embedded)  # [B, L, D_MODEL]

        # 4. Generate causal mask for the decoder's self-attention
        # Mask shape should be [L, L] where L is the target sequence length
        tgt_mask = self._generate_square_subsequent_mask(self.output_seq_len)  # [L, L]

        # 5. Prepare memory (encoded context) for the decoder's cross-attention
        # Although it's decoder-only, nn.TransformerDecoder expects 'memory'.
        # We feed the projected context here. It needs shape [S, B, E] where S=1.
        # projected_context: [B, D_MODEL] -> memory: [B, 1, D_MODEL]
        memory = projected_context.unsqueeze(1)

        # 6. Pass through Transformer Decoder
        # Input shape for batch_first=True: [B, L, D_MODEL]
        # Memory shape: [B, S, E] = [B, 1, D_MODEL]
        # Mask shape: [L, L]
        decoder_output = self.transformer_decoder(
            tgt=decoder_input_pos,  # Target sequence input [B, L, D_MODEL]
            memory=memory,  # Context from src [B, 1, D_MODEL]
            tgt_mask=tgt_mask,  # Causal mask [L, L]
            memory_mask=None  # No mask needed for memory attention here
        )
        # Output shape: [B, L, D_MODEL]

        # 7. Project to output dimension
        # output: [B, L, 1]
        output = self.output_projection(decoder_output)

        return output

    @torch.no_grad()  # Ensure no gradients are computed during inference
    def predict(self, src: torch.Tensor) -> torch.Tensor:
        """
        Inference/Prediction pass using auto-regressive generation.

        Args:
            src: Input tensor, shape [batch_size, input_dim (904)]

        Returns:
            Output tensor, shape [batch_size, output_seq_len, 1]
        """
        self.eval()  # Set model to evaluation mode
        batch_size = src.size(0)

        # 1. Project input context vector
        # src: [B, 904] -> projected_context: [B, D_MODEL]
        projected_context = self.input_projection(src)

        # 2. Prepare memory for decoder cross-attention
        # memory: [B, 1, D_MODEL]
        memory = projected_context.unsqueeze(1)

        # 3. Initialize the decoder input sequence
        # Start with the projected context as the first input token (representing step 0)
        # decoder_input_embedded: [B, 1, D_MODEL]
        decoder_input_embedded = projected_context.unsqueeze(1)

        # 4. Store the generated output sequence
        # Initialize with zeros or another placeholder if needed, size [B, L, 1]
        outputs = torch.zeros(batch_size, self.output_seq_len, 1, device=self._device)

        # 5. Auto-regressive generation loop
        for i in range(self.output_seq_len):
            # Get current sequence length
            current_seq_len = decoder_input_embedded.size(1)

            # Add positional encoding to the current input sequence
            decoder_input_pos = self.pos_encoder(decoder_input_embedded)  # [B, current_seq_len, D_MODEL]

            # Generate causal mask for the current length
            tgt_mask = self._generate_square_subsequent_mask(current_seq_len)  # [current_seq_len, current_seq_len]

            # Pass through Transformer Decoder
            # Input: [B, current_seq_len, D_MODEL]
            # Memory: [B, 1, D_MODEL]
            # Mask: [current_seq_len, current_seq_len]
            decoder_output = self.transformer_decoder(
                tgt=decoder_input_pos,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=None
            )
            # Output shape: [B, current_seq_len, D_MODEL]

            # Get the prediction for the *next* time step (last element of the sequence)
            # last_step_output: [B, D_MODEL]
            last_step_output = decoder_output[:, -1, :]

            # Project to output dimension
            # next_output_val: [B, 1]
            next_output_val = self.output_projection(last_step_output)

            # Store the prediction
            outputs[:, i, :] = next_output_val

            # Prepare the input for the *next* iteration
            # Embed the predicted value and append it to the decoder input sequence
            # next_output_embedded: [B, 1, D_MODEL]
            next_output_embedded = self.target_embedder(
                next_output_val.unsqueeze(-1))  # Add feature dim before embedding
            # decoder_input_embedded: [B, current_seq_len + 1, D_MODEL]
            decoder_input_embedded = torch.cat([decoder_input_embedded, next_output_embedded], dim=1)

        self.train()  # Set model back to training mode
        return outputs


class Transformer(nn.Module):
    """
    The main Transformer model that combines the Encoder and Decoder, using
    the built-in PyTorch `nn.Transformer` module.
    """


    def __init__(
            self,
            num_encoder_layers,
            num_decoder_layers,
            d_model,
            num_heads,
            d_ff,
            input_features,
            output_sequence_length,
            dropout_rate=0.1
    ):
        """
        Args:
            num_encoder_layers (int): Number of encoder layers.
            num_decoder_layers (int): Number of decoder layers.
            d_model (int): The dimensionality of the model's embeddings.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimensionality of the feed-forward network's inner layer.
            input_features (int): Number of features in the source/input data.
            output_sequence_length (int): Number of features in the target data.
            dropout_rate (float): The dropout probability.
        """
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.d_model_sqrt = math.sqrt(self.d_model)
        self.output_sequence_length = output_sequence_length

        # --- Layers ---
        self.src_projection = nn.Linear(input_features, d_model)
        self.tgt_projection = nn.Linear(1, d_model)  # Projects univariate target series
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate)

        # Unified Transformer module
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_ff,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
            # to stabilize training while applying stronger regularization
            # c.f. https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8/
            layer_norm_eps=1e-2
        )

        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        """
        Forward pass for training (using teacher forcing).

        This method internally creates the decoder input by prepending a
        start token to the target sequence, making the training loop cleaner.

        Args:
            src (torch.Tensor): The source vector. Shape: [batch_size, input_dim].
            tgt (torch.Tensor): The ground-truth target sequence. Shape: [batch_size, output_sequence_length].

        Returns:
            torch.Tensor: The model's prediction. Shape: [batch_size, output_sequence_length].
        """
        # 1. Get parameters from input tensors
        device = tgt.device
        batch_size = tgt.size(0)

        # 2. Create the decoder input sequence by prepending a start token (0)
        # Shape: [B, 1]
        start_token = torch.zeros(batch_size, 1, device=device)
        # Shape: [B, 431] (all but the last element of the target sequence)
        tgt_shifted_right = tgt[:, :-1]
        # Shape: [B, 432] (concatenated for the final decoder input)
        decoder_input = torch.cat((start_token, tgt_shifted_right), dim=1)

        # 3. Reshape, project, and apply positional encoding
        src = self.src_projection(src.unsqueeze(1)) * self.d_model_sqrt
        decoder_input = self.tgt_projection(decoder_input.unsqueeze(-1)) * self.d_model_sqrt
        decoder_input = self.pos_encoder(decoder_input)

        # 4. Pass through the Transformer (causal mask is generated automatically)
        # output = self.transformer(src, decoder_input, tgt_is_causal=True)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.size(1)).to(device)
        memory = self.transformer.encoder(src)
        output = self.transformer.decoder(decoder_input, memory, tgt_mask=tgt_mask, tgt_is_causal=True)

        # 5. Final output projection
        # The output shape is [B, 432, 1], which we squeeze to [B, 432]
        predictions = self.output_layer(output)
        return predictions.squeeze(-1)

    @torch.jit.export
    @torch.no_grad()
    def predict(self, src: torch.Tensor) -> torch.Tensor:
        """
        Inference function for autoregressive prediction.

        Args:
            src (torch.Tensor): The source vector. Shape: [batch_size, input_dim].

        Returns:
            torch.Tensor: The generated output sequence. Shape: [batch_size, output_sequence_length].
        """
        self.eval()  # Set model to evaluation mode
        device = src.device
        batch_size = src.size(0)

        # 1. Encode the source vector once
        # src: [B, 98] -> [B, 1, 98] -> [B, 1, D_MODEL]
        src_projected = self.src_projection(src.unsqueeze(1)) * self.d_model_sqrt
        memory = self.transformer.encoder(src_projected)  # Shape: [B, 1, D_MODEL]

        # 2. Initialize the target sequence with a start token (e.g., zeros)
        # It starts as [B, 1] and grows to [B, output_sequence_length]
        generated_sequence = torch.zeros(batch_size, 1 + self.output_sequence_length, device=device)

        # 3. Autoregressive generation loop
        for i in range(self.output_sequence_length):
            # Project and encode the current generated sequence
            # gen_seq: [B, T] -> [B, T, 1] -> [B, T, D_MODEL]
            tgt_in = self.tgt_projection(generated_sequence[:, :i + 1].unsqueeze(-1)) * self.d_model_sqrt
            tgt_in = self.pos_encoder(tgt_in)

            # Generate a mask for the current sequence length
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(device)

            # Get the model's prediction for the next step
            output = self.transformer.decoder(tgt_in, memory, tgt_mask=tgt_mask, tgt_is_causal=True)

            # Take the output from the last time step and project it to a single value
            next_val = self.output_layer(output[:, -1:, :])  # Shape: [B, 1, 1]

            # Append the prediction to our generated sequence
            generated_sequence[:, i + 1] = next_val.reshape(-1)

        # Return the generated sequence, excluding the initial start token
        return generated_sequence[:, 1:]
