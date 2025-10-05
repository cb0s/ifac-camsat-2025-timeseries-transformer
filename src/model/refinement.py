# --- Refinement Transformer (Encoder-Decoder) ---
import torch
from torch import nn

from src.model.encoder import PositionalEncodingDecoderOnly


class RefinementTransformer(nn.Module):
    def __init__(self,
                 initial_pred_seq_len: int,
                 additional_context_dim: int,
                 output_seq_len: int,
                 d_model: int,
                 n_heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 d_ff: int,
                 dropout: float = 0.1,
                 max_output_len: int = 500, # For pos encoding buffer size
                 device='cuda'):
        super().__init__()
        self.initial_pred_seq_len = initial_pred_seq_len
        self.output_seq_len = output_seq_len
        self.d_model = d_model

        # --- Embeddings ---
        # 1. Embed the initial prediction sequence (1 dim per step -> d_model)
        self.initial_pred_embedder = nn.Linear(1, d_model)
        # 2. Embed the static additional context features (104 dim -> d_model)
        self.context_embedder = nn.Linear(additional_context_dim, d_model)
        # 3. Embed the target sequence for the decoder input (1 dim per step -> d_model)
        self.target_embedder = nn.Linear(1, d_model)

        # --- Positional Encoding ---
        self.pos_encoder = PositionalEncodingDecoderOnly(d_model, dropout, max_len=max_output_len)

        # --- Start Of Sequence (SOS) Token for Decoder ---
        # Learnable embedding for the first input token to the decoder
        self.decoder_sos_token = nn.Parameter(torch.randn(1, 1, d_model))

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-LN
        )
        encoder_norm = nn.LayerNorm(d_model) # Optional norm after stack
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm
        )

        # --- Transformer Decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-LN
        )
        decoder_norm = nn.LayerNorm(d_model) # Optional norm after stack
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            norm=decoder_norm
        )

        # --- Output Layer ---
        # Project decoder output (d_model) to 1 dimension (refined value)
        self.output_projection = nn.Linear(d_model, 1)

        self._init_weights()

        self._device = torch.device(device)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Optional: Initialize SOS token differently if needed
        # nn.init.normal_(self.decoder_sos_token, mean=0., std=d_model**-0.5)

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.empty(sz, sz, device=self._device).fill_(float('-inf')), diagonal=1)

    def forward(self,
                initial_predictions: torch.Tensor, # Shape: [B, 432, 1]
                context_features: torch.Tensor,   # Shape: [B, 104]
                target_sequence: torch.Tensor     # Shape: [B, 432, 1] (Ground truth refined)
    ) -> torch.Tensor:
        """
        Training pass using Teacher Forcing.

        Args:
            initial_predictions: The sequence output from the first model.
            context_features: The additional static features.
            target_sequence: The ground truth refined sequence for teacher forcing.

        Returns:
            Output tensor (refined predictions), shape [batch_size, output_seq_len, 1]
        """
        batch_size = initial_predictions.size(0)

        # --- Encoder Input Preparation ---
        # 1. Embed initial predictions: [B, 432, 1] -> [B, 432, D]
        embedded_preds = self.initial_pred_embedder(initial_predictions)
        # 2. Embed context: [B, 104] -> [B, D]
        embedded_context = self.context_embedder(context_features)
        # 3. Combine: Add context embedding to each timestep of the sequence embedding
        # Unsqueeze context: [B, D] -> [B, 1, D]
        encoder_input_base = embedded_preds + embedded_context.unsqueeze(1)
        # 4. Add positional encoding: [B, 432, D] -> [B, 432, D]
        encoder_input = self.pos_encoder(encoder_input_base)

        # --- Encoder Pass ---
        # No padding mask needed as input sequence length is fixed (432)
        # encoder_output shape: [B, 432, D]
        encoder_output = self.transformer_encoder(encoder_input, mask=None)

        # --- Decoder Input Preparation (Teacher Forcing) ---
        # 1. Embed target sequence (shifted right): [B, 432, 1] -> [B, 431, 1] -> [B, 431, D]
        embedded_target = self.target_embedder(target_sequence[:, :-1, :])
        # 2. Prepend SOS token: SOS shape [1, 1, D], expand to [B, 1, D]
        sos_token = self.decoder_sos_token.expand(batch_size, -1, -1)
        # Concatenate: [B, 1, D] + [B, 431, D] -> [B, 432, D]
        decoder_input_base = torch.cat([sos_token, embedded_target], dim=1)
        # 3. Add positional encoding: [B, 432, D] -> [B, 432, D]
        decoder_input = self.pos_encoder(decoder_input_base)

        # --- Decoder Pass ---
        # 1. Generate causal mask for decoder self-attention: [432, 432]
        tgt_mask = self._generate_square_subsequent_mask(self.output_seq_len)
        # 2. Pass through decoder: Uses decoder_input, encoder_output (as memory), and tgt_mask
        # decoder_output shape: [B, 432, D]
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=encoder_output,
            tgt_mask=tgt_mask,
            memory_mask=None # No mask needed for memory attention here
        )

        # --- Output Projection ---
        # Project decoder output to final dimension: [B, 432, D] -> [B, 432, 1]
        final_output = self.output_projection(decoder_output)

        return final_output

    @torch.no_grad()
    def predict(self,
                initial_predictions: torch.Tensor, # Shape: [B, 432, 1]
                context_features: torch.Tensor    # Shape: [B, 104]
                ) -> torch.Tensor:
        """
        Inference/Prediction pass using auto-regressive generation.

        Args:
            initial_predictions: The sequence output from the first model.
            context_features: The additional static features.

        Returns:
            Output tensor (refined predictions), shape [batch_size, output_seq_len, 1]
        """
        self.eval() # Set model to evaluation mode
        batch_size = initial_predictions.size(0)

        # --- Encode Input Sequence and Context ---
        # Same preparation as in forward pass
        embedded_preds = self.initial_pred_embedder(initial_predictions)
        embedded_context = self.context_embedder(context_features)
        encoder_input_base = embedded_preds + embedded_context.unsqueeze(1)
        encoder_input = self.pos_encoder(encoder_input_base)
        encoder_output = self.transformer_encoder(encoder_input) # [B, 432, D]

        # --- Auto-regressive Decoding ---
        # Start with SOS token
        decoder_input_embedded = self.decoder_sos_token.expand(batch_size, -1, -1) # [B, 1, D]
        outputs = torch.zeros(batch_size, self.output_seq_len, 1, device=self._device)

        for i in range(self.output_seq_len):
            current_seq_len = decoder_input_embedded.size(1)

            # Add positional encoding
            decoder_input_pos = self.pos_encoder(decoder_input_embedded) # [B, current_seq_len, D]

            # Generate causal mask for current length
            tgt_mask = self._generate_square_subsequent_mask(current_seq_len)

            # Pass through decoder
            decoder_output = self.transformer_decoder(
                tgt=decoder_input_pos,
                memory=encoder_output,
                tgt_mask=tgt_mask
            ) # [B, current_seq_len, D]

            # Get the prediction for the *next* time step (last element)
            last_step_output = decoder_output[:, -1, :] # [B, D]

            # Project to output dimension
            next_output_val = self.output_projection(last_step_output) # [B, 1]

            # Store prediction
            outputs[:, i, :] = next_output_val

            # Prepare input for the *next* iteration if not the last step
            if i < self.output_seq_len - 1:
                # Embed the predicted value: [B, 1] -> [B, 1, 1] -> [B, 1, D]
                next_output_embedded = self.target_embedder(next_output_val.unsqueeze(-1))
                # Append to the current decoder input sequence: [B, current_seq_len + 1, D]
                decoder_input_embedded = torch.cat([decoder_input_embedded, next_output_embedded], dim=1)

        self.train() # Set model back to training mode
        return outputs
