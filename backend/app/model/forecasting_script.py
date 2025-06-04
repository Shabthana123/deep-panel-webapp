import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.data import TimeSeriesDataSet, MultiNormalizer, GroupNormalizer
from datetime import datetime, timedelta
from pytorch_forecasting.metrics import QuantileLoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import lightning.pytorch as pl
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    InterpretableMultiHeadAttention, GateAddNorm, GatedResidualNetwork
)
from typing import Dict, List, Optional, Tuple, Union
import math
from pathlib import Path
import warnings
import argparse
warnings.filterwarnings("ignore")

# Define paths
# BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = Path.cwd() 
print(f"Base directory: {BASE_DIR}")

# Dataset configurations
DATASET_CONFIGS = {
    'covid_death': {
        'data_file': 'covid_death.csv',
        'model_file': 'covid_death.ckpt',
        'time_idx_type': 'daily',
        'columns': ['Entity', 'Date', 'Value'],
        'freq_offset': 'D',
        'dataset_name': 'Covid Deaths',
        'type':'Unbalanced, Macro, Long'
    },
    'covid_confirmed': {
        'data_file': 'covid_confirmed.csv',
        'model_file': 'covid_confirmed.ckpt',
        'time_idx_type': 'daily',
        'columns': ['Entity', 'Date', 'Value'],
        'freq_offset': 'D',
        'dataset_name': 'Covid Confirmed',
        'type':'Unbalanced, Macro, Long'
    },
    'covid_recovered': {
        'data_file': 'covid_recovered.csv',
        'model_file': 'covid_recovered.ckpt',
        'time_idx_type': 'daily',
        'columns': ['Entity', 'Date', 'Value'],
        'freq_offset': 'D',
        'dataset_name': 'Covid Recovered',
        'type':'Unbalanced, Macro, Long'
    },
    'african_GDP': {
        'data_file': 'african_GDP.csv',
        'model_file': 'african_GDP.ckpt',
        'time_idx_type': 'annual',
        'columns': ['Entity', 'Date', 'Value', 'GNI', 'PPP'],
        'freq_offset': 'Y1',
        'dataset_name': 'GDP Africa',
        'type':'Balanced, Macro, Short'
    },
    'co2': {
        'data_file': 'co2.csv',
        'model_file': 'co2.ckpt',
        'time_idx_type': 'annual',
        'columns': ['Entity', 'Date', 'Value'],
        'freq_offset': 'Y1',
        'dataset_name': 'CO2',
        'type':'Unbalanced, Macro, Short'
    },
    'air_traffic': {
        'data_file': 'air_traffic.csv',
        'model_file': 'air_traffic.ckpt',
        'time_idx_type': 'monthly',
        'columns': ['Entity', 'Date', 'Value'],
        'freq_offset': 'M',
        'dataset_name': 'Traffic',
        'type':'Balanced, Micro, Long'
    },
    'sales': {
        'data_file': 'sales.csv',
        'model_file': 'sales.ckpt',
        'time_idx_type': 'monthly',
        'columns': ['Entity', 'Date', 'Value'],
        'freq_offset': 'M',
        'dataset_name': 'Sales',
        'type':'Balanced, Micro, Short'
    },
    'exchange_rate': {
        'data_file': 'exchange_rate.csv',
        'model_file': 'exchange_rate.ckpt',
        'time_idx_type': 'daily',
        'columns': ['Entity', 'Date', 'Value'],
        'freq_offset': 'D',
        'dataset_name': 'Exchange Rate',
        'type':'Balanced, Macro, Long'
    },
    'surface_temperature': {
        'data_file': 'surface_temperature.csv',
        'model_file': 'surface_temperature.ckpt',
        'time_idx_type': 'monthly',
        'columns': ['Entity', 'Date', 'Value'],
        'freq_offset': 'M',
        'dataset_name': 'Temperature',
        'type':'Balanced, Macro, Long'
    },
    'stock': {
        'data_file': 'stock.csv',
        'model_file': 'stock.ckpt',
        'time_idx_type': 'annual',
        'columns': ['Entity', 'Date', 'Value'],
        'freq_offset': 'Y',
        'dataset_name': 'Stock',
        'type':'Unbalanced, Micro, Short'
    },
    'electricity': {
        'data_file': 'electricity.csv',
        'model_file': 'electricity.ckpt',
        'time_idx_type': 'daily',
        'columns': ['Entity', 'Date', 'Value'],
        'freq_offset': 'D',
        'dataset_name': 'Electricity',
        'type':'Unbalanced, Micro, Long'
    }
}

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.softmax = nn.Softmax(dim=-1)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        # Compute attention scores
        attn = torch.bmm(q, k.permute(0, 2, 1))  # Query-key overlap

        if self.scale:
            dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension

        # Apply causal mask
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        # Apply softmax and dropout
        attn = self.softmax(attn)
        if self.dropout is not None:
            attn = self.dropout(attn)

        # Compute weighted sum of values
        output = torch.bmm(attn, v)
        return output, attn


class SegmentwiseInterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, segment_size: int, dropout: float = 0.0):
        super(SegmentwiseInterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.segment_size = segment_size
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        # Projection layer for skip connection
        self.skip_projection = nn.Linear(d_model, d_model)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, q_len, _ = q.shape
        _, kv_len, _ = k.shape

        # Dynamically compute number of segments
        num_full_segments = kv_len // self.segment_size
        remainder = kv_len % self.segment_size

        if remainder > 0:
            # Add a smaller segment for the remainder
            num_segments = num_full_segments + 1
        else:
            num_segments = num_full_segments

        # Reshape keys and values into segments
        k_segments = []
        v_segments = []
        mask_segments = []  # To store masks for each segment

        for i in range(num_segments):
            start_idx = i * self.segment_size
            end_idx = min((i + 1) * self.segment_size, kv_len)
            k_seg = k[:, start_idx:end_idx, :]  # Shape: [batch_size, segment_len, d_model]
            v_seg = v[:, start_idx:end_idx, :]

            # Pad shorter segments to match segment_size
            if k_seg.size(1) < self.segment_size:
                pad_len = self.segment_size - k_seg.size(1)
                k_seg = F.pad(k_seg, (0, 0, 0, pad_len))  # Pad along the sequence length dimension
                v_seg = F.pad(v_seg, (0, 0, 0, pad_len))

            k_segments.append(k_seg)
            v_segments.append(v_seg)

            # Handle the mask for this segment
            if mask is not None:
                mask_seg = mask[:, :, start_idx:end_idx]  # Extract the mask for this segment
                if mask_seg.size(-1) < self.segment_size:
                    pad_len = self.segment_size - mask_seg.size(-1)
                    mask_seg = F.pad(mask_seg, (0, pad_len), value=True)  # Pad with True (masked)
                mask_segments.append(mask_seg)

        # Stack segments into single tensors
        k_segments = torch.stack(k_segments, dim=1)  # Shape: [batch_size, num_segments, segment_size, d_model]
        v_segments = torch.stack(v_segments, dim=1)  # Shape: [batch_size, num_segments, segment_size, d_model]
        if mask is not None:
            mask_segments = torch.stack(mask_segments, dim=1)  # Shape: [batch_size, num_segments, q_len, segment_size]

        heads = []
        attns = []

        vs = self.v_layer(v)  # Apply linear projection to values

        for i in range(self.n_head):
            qs = self.q_layers[i](q)  # Project queries
            head_segments = []
            attn_segments = []

            for seg_idx in range(num_segments):
                # Extract the current segment of keys, values, and mask
                k_seg = k_segments[:, seg_idx, :, :]  # Shape: [batch_size, segment_size, d_model]
                v_seg = v_segments[:, seg_idx, :, :]
                mask_seg = mask_segments[:, seg_idx, :, :] if mask is not None else None

                # Ensure k_seg has the correct shape for the linear layer
                assert k_seg.size(-1) == self.d_model, f"k_seg last dimension ({k_seg.size(-1)}) must match d_model ({self.d_model})"

                # Create causal mask for this segment
                if mask_seg is None:
                    causal_mask = torch.triu(
                        torch.ones(q_len, self.segment_size, dtype=torch.bool, device=q.device),
                        diagonal=1
                    ).unsqueeze(0)  # Shape: [1, q_len, segment_size]
                else:
                    causal_mask = mask_seg | torch.triu(
                        torch.ones(q_len, self.segment_size, dtype=torch.bool, device=q.device),
                        diagonal=1
                    ).unsqueeze(0)  # Combine causal mask with existing mask

                # Compute attention for this segment
                k_seg_proj = self.k_layers[i](k_seg)  # Project keys for this segment
                v_seg_proj = self.v_layer(v_seg)  # Project values for this segment
                head_seg, attn_seg = self.attention(qs, k_seg_proj, v_seg_proj, mask=causal_mask)

                head_segments.append(head_seg)
                attn_segments.append(attn_seg)

            # Concatenate attention outputs from all segments
            head = torch.cat(head_segments, dim=1)
            attn = torch.cat(attn_segments, dim=1)

            # Apply dropout to the concatenated head
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        # Stack outputs from all heads
        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        # Average over heads and apply final linear layer
        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)
        return outputs, attn

class MultiScaleMovingAvg(nn.Module):
    def __init__(self, kernel_sizes=[3, 7, 15, 31]):
        super(MultiScaleMovingAvg, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.n_kernels = len(kernel_sizes)

        # Create a learnable weight for each kernel size
        self.kernel_weights = nn.Parameter(torch.ones(self.n_kernels) / self.n_kernels)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # x shape: (batch, time, features)
        batch_size, seq_len, n_features = x.shape
        x_permuted = x.permute(0, 2, 1)  # (batch, features, time)

        # Apply moving average with different kernel sizes
        ma_outputs = []
        for k_size in self.kernel_sizes:
            # Create kernel
            kernel = torch.ones(1, 1, k_size, device=x.device) / k_size
            kernel = kernel.repeat(n_features, 1, 1)  # (features, 1, kernel_size)

            # Apply padding
            pad_size = k_size // 2
            x_padded = F.pad(x_permuted, pad=(pad_size, pad_size), mode='replicate')

            # Apply convolution
            ma = F.conv1d(x_padded, kernel, groups=n_features)
            ma_outputs.append(ma.permute(0, 2, 1))  # Back to (batch, time, features)

        # Compute adaptive weights
        weights = self.softmax(self.kernel_weights)

        # Weighted sum of moving averages
        ma_combined = torch.zeros_like(x)
        for i, ma in enumerate(ma_outputs):
            ma_combined += weights[i] * ma
            print('ma_combined: ',ma_combined, '\n')

        return ma_combined

class EnhancedSeriesDecomp(nn.Module):
    def __init__(self, kernel_sizes=[3, 7, 15, 31]):
        super(EnhancedSeriesDecomp, self).__init__()
        self.moving_avg = MultiScaleMovingAvg(kernel_sizes)

    def forward(self, x):
        # Extract trend using multi-scale moving average
        trend = self.moving_avg(x)
        # Extract seasonal component (residual)
        seasonal = x - trend
        return seasonal, trend

def compute_component_weights(seasonal, trend):
    # Compute variance along the time dimension
    seasonal_var = torch.var(seasonal, dim=1, keepdim=True)  # Shape: (batch, 1, features)
    trend_var = torch.var(trend, dim=1, keepdim=True)        # Shape: (batch, 1, features)

    # Normalize variances to get weights
    total_var = seasonal_var + trend_var
    seasonal_weight = seasonal_var / (total_var + 1e-8)  # Avoid division by zero
    trend_weight = trend_var / (total_var + 1e-8)

    return seasonal_weight, trend_weight

def compute_loss(model_output, ground_truth, target_scale, trend_weight=0.1):
    # Extract predictions
    prediction = model_output["prediction"]
    trend_prediction = model_output["trend_prediction"]

    # Inverse transform
    prediction = prediction * target_scale
    ground_truth = ground_truth * target_scale

    # Decompose ground truth
    seasonal_ground_truth, trend_ground_truth = model.decomposition(ground_truth)

    # Compute main prediction loss
    prediction_loss = F.mse_loss(prediction, ground_truth)

    # Compute trend loss
    trend_loss = F.mse_loss(trend_prediction, trend_ground_truth)

    # Combine losses
    total_loss = prediction_loss + trend_weight * trend_loss
    return total_loss

class CrossSeriesAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        assert hidden_size % num_heads == 0, "embed_dim must be divisible by n_heads"
        self.d_k = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        
        # Add dropout layer for attention probabilities
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        if context.dim() == 2:
            context = context.unsqueeze(1)  # (B, 1, H)

        B, T, _ = query.size()
        _, S, _ = context.size()

        Q = self.q_linear(query)  # (B, T, H)
        K = self.k_linear(context)  # (B, S, H)
        V = self.v_linear(context)  # (B, S, H)

        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
        K = K.view(B, S, self.num_heads, self.d_k).transpose(1, 2)  # (B, n_heads, S, d_k)
        V = V.view(B, S, self.num_heads, self.d_k).transpose(1, 2)  # (B, n_heads, S, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)  # apply dropout here

        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous().view(B, T, self.hidden_size)
        output = self.out_linear(output)
        return output
        
class EnhancedTFT(TemporalFusionTransformer):
    def __init__(
        self, 
        *args, 
        segment_size=8,
        decomposition_kernel_sizes=[3, 7, 15, 31],
        trend_processing_layers=2,
        use_cross_series_attention=True,
        adaptive_trend_weight=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Store new hyperparameters
        self.hparams.segment_size = segment_size
        self.hparams.decomposition_kernel_sizes = decomposition_kernel_sizes
        self.hparams.trend_processing_layers = trend_processing_layers
        self.hparams.use_cross_series_attention = use_cross_series_attention
        self.hparams.adaptive_trend_weight = adaptive_trend_weight

        # 1. Replace standard attention with segmentwise attention
        self.multihead_attn = SegmentwiseInterpretableMultiHeadAttention(
            n_head=self.hparams.attention_head_size,
            d_model=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            segment_size=segment_size
        )

        # 2. Add enhanced series decomposition
        self.decomposition = EnhancedSeriesDecomp(kernel_sizes=decomposition_kernel_sizes)

        # 3. Add trend processing network
        trend_layers = []
        for _ in range(trend_processing_layers):
            trend_layers.append(
                GatedResidualNetwork(
                    input_size=self.hparams.hidden_size,
                    hidden_size=self.hparams.hidden_size,
                    output_size=self.hparams.hidden_size,
                    dropout=self.hparams.dropout
                )
            )
        self.trend_processor = nn.Sequential(*trend_layers)

        # 4. Add cross-series attention if enabled
        if use_cross_series_attention:
            self.cross_series_attn = CrossSeriesAttention(
                hidden_size=self.hparams.hidden_size,
                num_heads=self.hparams.attention_head_size,
                dropout=self.hparams.dropout
            )

        # 5. Add adaptive trend weight network if enabled
        if adaptive_trend_weight:
            self.trend_weight_network = nn.Sequential(
                nn.Linear(self.hparams.hidden_size * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        # 6. Modify output layer to handle combined seasonal and trend components
        if self.n_targets > 1:
            self.output_layer = nn.ModuleList(
                [
                    nn.Linear(self.hparams.hidden_size * 2, output_size)
                    for output_size in self.hparams.output_size
                ]
            )
        else:
            self.output_layer = nn.Linear(
                self.hparams.hidden_size * 2, self.hparams.output_size
            )

    def forward(self, x):
        # Get standard inputs
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)
        timesteps = x_cont.size(1)
        max_encoder_length = int(encoder_lengths.max())

        # Standard TFT embedding and variable selection
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update({
            name: x_cont[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.hparams.x_reals)
            if name in self.reals
        })

        # Static embedding
        if len(self.static_variables) > 0:
            static_embedding = {
                name: input_vectors[name][:, 0] for name in self.static_variables
            }
            static_embedding, static_variable_selection = (
                self.static_variable_selection(static_embedding)
            )
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            static_variable_selection = torch.zeros(
                (x_cont.size(0), 0), dtype=self.dtype, device=self.device
            )

        # Variable selection with static context
        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        # Encoder variable selection
        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length]
            for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = (
            self.encoder_variable_selection(
                embeddings_varying_encoder,
                static_context_variable_selection[:, :max_encoder_length],
            )
        )

        # Decoder variable selection
        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:]
            for name in self.decoder_variables
        }
        embeddings_varying_decoder, decoder_sparse_weights = (
            self.decoder_variable_selection(
                embeddings_varying_decoder,
                static_context_variable_selection[:, max_encoder_length:],
            )
        )

        # Series decomposition after variable selection
        seasonal_enc, trend_enc = self.decomposition(embeddings_varying_encoder)
        seasonal_dec, trend_dec = self.decomposition(embeddings_varying_decoder)

        # LSTM for seasonal component
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )

        encoder_output, (hidden, cell) = self.lstm_encoder(
            seasonal_enc,
            (input_hidden, input_cell),
            lengths=encoder_lengths,
            enforce_sorted=False,
        )
        decoder_output, _ = self.lstm_decoder(
            seasonal_dec,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, seasonal_enc)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, seasonal_dec)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # Static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output,
            self.expand_static_context(static_context_enrichment, timesteps),
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # Query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_lengths=decoder_lengths
            ),
        )

        # Cross-series attention if enabled
        if hasattr(self, 'cross_series_attn'):
            attn_output = self.cross_series_attn(attn_output, static_embedding)

        # Skip connections
        skip_tensor = attn_input[:, max_encoder_length:]
        min_len = min(attn_output.shape[1], skip_tensor.shape[1])
        attn_output = attn_output[:, :min_len, :]
        skip_tensor = skip_tensor[:, :min_len, :]
        attn_output = self.post_attn_gate_norm(attn_output, skip_tensor)

        seasonal_output = self.pos_wise_ff(attn_output)

        skip_tensor = lstm_output[:, max_encoder_length:]
        min_len = min(seasonal_output.shape[1], skip_tensor.shape[1])
        seasonal_output = seasonal_output[:, :min_len, :]
        skip_tensor = skip_tensor[:, :min_len, :]
        seasonal_output = self.pre_output_gate_norm(seasonal_output, skip_tensor)

        # Process trend component
        trend_all = torch.cat([trend_enc, trend_dec], dim=1)
        trend_output = trend_all[:, max_encoder_length:]  # Only decoder part
        processed_trend = self.trend_processor(trend_output)

        # Dynamically weigh seasonal and trend components
        if hasattr(self, 'trend_weight_network'):
            # Use learned weights
            combined_features = torch.cat([seasonal_output, processed_trend], dim=-1)
            trend_weight = self.trend_weight_network(combined_features)
            seasonal_weight = 1 - trend_weight

            weighted_seasonal = seasonal_output * seasonal_weight
            weighted_trend = processed_trend * trend_weight
        else:
            # Use variance-based weights
            seasonal_weight, trend_weight = compute_component_weights(seasonal_output, processed_trend)
            weighted_seasonal = seasonal_output * seasonal_weight
            weighted_trend = processed_trend * trend_weight

        # Combine components for final prediction
        combined_output = torch.cat([weighted_seasonal, weighted_trend], dim=-1)

        # Final output layer
        if self.n_targets > 1:
            output = [output_layer(combined_output) for output_layer in self.output_layer]
        else:
            output = self.output_layer(combined_output)

        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"]),
            trend_prediction=processed_trend,
            encoder_attention=attn_output_weights[..., :max_encoder_length],
            decoder_attention=attn_output_weights[..., max_encoder_length:],
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )



def get_time_idx(data, freq_offset):
    """Create time index based on the dataset type"""
    if freq_offset == 'D':
        # Convert to datetime format
        data["Date"] = pd.to_datetime(data["Date"])
        return (data['Date'] - data['Date'].min()).dt.days
    elif freq_offset == 'M':
        # Convert to datetime format
        data["Date"] = pd.to_datetime(data["Date"])
        return ((data["Date"].dt.year - data["Date"].dt.year.min()) * 12) + (data["Date"].dt.month - data["Date"].dt.month.min())
    elif freq_offset == 'Y':
        # Convert to datetime format
        data["Date"] = pd.to_datetime(data["Date"])
        return data["Date"].dt.year - data["Date"].dt.year.min()
    elif freq_offset == 'Y1':
        data['Date'] = pd.to_datetime(data['Date'], format='%Y')
        return data['Date'].dt.year - data['Date'].min().year
        
        # return (data["Date"] - data["Date"].min())
    else:
        raise ValueError(f"Unknown time_idx_type: {freq_offset}")


def load_and_preprocess_data(dataset_key):
    """Load and preprocess data based on dataset configuration"""
    config = DATASET_CONFIGS[dataset_key]
    data_path = BASE_DIR / "data" / config['data_file']
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    data = pd.read_csv(data_path)
    
    # Set column names based on configuration
    if len(config['columns']) == 3:
        data.columns = ['Entity', 'Date', 'Value']
    elif len(config['columns']) == 5:  # For african_GDP
        data.columns = ['Entity', 'Date', 'Value', 'GNI', 'PPP']
    
      
    data['Value'] = data['Value'].astype(float)
    
    # Handle special cases
    if dataset_key == 'sales':
        data["Entity"] = data["Entity"].astype(str)
    
    # Create time index based on dataset type
    data["time_idx"] = get_time_idx(data, config['freq_offset'])
    
    return data, config


def create_forecast_dataset(data, config, max_encoder_length=30):
    """Create dataset for forecasting"""
    # Use the same parameters as in training
    total_time_steps = data['time_idx'].max() + 1
    train_size = 0.7
    training_cutoff = int(total_time_steps * train_size)
    
    # Create training dataset (needed for dataset structure)
    training = TimeSeriesDataSet(
        data[lambda x: x['time_idx'] <= training_cutoff],
        time_idx='time_idx',
        target='Value',
        group_ids=['Entity'],
        max_encoder_length=max_encoder_length,
        min_encoder_length=max_encoder_length // 2,
        max_prediction_length=30,
        min_prediction_length=1,
        static_categoricals=['Entity'],
        time_varying_unknown_reals=['Value'],
        time_varying_known_reals=['time_idx'],
        add_relative_time_idx=True,
        add_target_scales=True,
        target_normalizer=GroupNormalizer(groups=['Entity'], transformation='softplus')
    )
    
    return training


def create_future_data(data, config, steps_ahead=30):
    """Create future data points for forecasting"""
    future_data_list = []
    
    for entity in data['Entity'].unique():
        entity_data = data[data['Entity'] == entity].copy()
        entity_data = entity_data.sort_values('time_idx')
        
        # Get the last known values
        last_time_idx = entity_data['time_idx'].max()
        last_date = entity_data['Date'].max()
        last_value = entity_data['Value'].iloc[-1]
        
        # Create future time steps based on frequency
        for i in range(1, steps_ahead + 1):
            future_time_idx = last_time_idx + i
            
            if config['freq_offset'] == 'D':
                future_date = last_date + pd.DateOffset(days=i)
            elif config['freq_offset'] == 'M':
                future_date = last_date + pd.DateOffset(months=i)
            elif config['freq_offset'] == 'Y':
                future_date = last_date + pd.DateOffset(years=i)
            elif config['freq_offset'] == 'Y1':
                # future_date = last_date + i
                future_date = last_date + pd.DateOffset(years=i)
            
            future_row = {
                'Entity': entity,
                'Date': future_date,
                'Value': last_value,  # Placeholder value
                'time_idx': future_time_idx
            }
            future_data_list.append(future_row)
    
    future_df = pd.DataFrame(future_data_list)
    
    # Combine historical and future data
    extended_data = pd.concat([data, future_df], ignore_index=True)
    extended_data = extended_data.sort_values(['Entity', 'time_idx'])
    
    return extended_data

def forecast_next_30_steps(dataset_key):
    """Main forecasting function with quantile predictions"""
    print(f"Starting forecasting process for {dataset_key}...")
    
    # Get paths
    config = DATASET_CONFIGS[dataset_key]
    model_path = BASE_DIR / "checkpoints" / config['model_file']
    print(f"Model path: {model_path}")
    
    # Check if model file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    data, config = load_and_preprocess_data(dataset_key)
    print(f"Loaded data with {len(data)} rows and {data['Entity'].nunique()} entities")
    
    # Create dataset structure
    print("Creating dataset structure...")
    training_dataset = create_forecast_dataset(data, config)
    
    # Load the trained model
    print("Loading trained model...")
    model = EnhancedTFT.load_from_checkpoint(
        model_path,
        map_location='cpu'  # Use CPU for inference
    )
    model.eval()
    
    # Create extended data with future time points
    print("Creating future data points...")
    extended_data = create_future_data(data, config, steps_ahead=30)
    
    # Get the latest data for each entity (this will be used as encoder input)
    predictions_list = []
    
    for entity in data['Entity'].unique():
        print(f"Forecasting for entity: {entity}")
        
        entity_data = extended_data[extended_data['Entity'] == entity].copy()
        entity_data = entity_data.sort_values('time_idx')
        
        # Get the last 30 points as encoder input and next 30 as decoder target
        total_points = len(entity_data)
        encoder_end = total_points - 30  # Use historical data for encoding
        
        if encoder_end < 30:
            print(f"Warning: Not enough historical data for entity {entity}")
            continue
        
        # Create forecast dataset for this entity
        forecast_data = entity_data.iloc[encoder_end-30:].copy()  # Last 60 points (30 historical + 30 future)
        
        try:
            # Create dataset for prediction
            forecast_dataset = TimeSeriesDataSet.from_dataset(
                training_dataset, 
                forecast_data, 
                predict=True, 
                stop_randomization=True
            )
            
            # Create dataloader
            forecast_dataloader = forecast_dataset.to_dataloader(
                train=False, 
                batch_size=1, 
                num_workers=0
            )
            
            # Make predictions using raw mode to get quantiles
            with torch.no_grad():
                raw_predictions = model.predict(forecast_dataloader, mode="raw", return_x=True)
            
            # Extract predictions and quantiles
            predictions_output = raw_predictions.output
            x_data = raw_predictions.x
            
            # Get quantile predictions
            if hasattr(predictions_output, 'prediction'):
                # For newer versions of pytorch-forecasting
                quantile_predictions = predictions_output.prediction
            else:
                # For older versions
                quantile_predictions = predictions_output
            
            # Convert to numpy
            if isinstance(quantile_predictions, torch.Tensor):
                quantile_predictions = quantile_predictions.cpu().numpy()
            
            # Get quantile levels from the model (your specific quantiles)
            quantiles = model.loss.quantiles if hasattr(model.loss, 'quantiles') else [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
            print("quantiles", quantiles)
            
            # Handle different tensor shapes
            if len(quantile_predictions.shape) == 3:
                # Shape: (batch_size, sequence_length, num_quantiles)
                quantile_predictions = quantile_predictions[0]  # Take first batch
            elif len(quantile_predictions.shape) == 2:
                # Shape: (sequence_length, num_quantiles)
                pass
            else:
                # If 1D, assume it's median prediction
                quantile_predictions = quantile_predictions.reshape(-1, 1)
                quantiles = [0.5]
            
            # Create future dates
            if config['freq_offset'] == 'D':
                future_dates = pd.date_range(
                    start=data[data['Entity'] == entity]['Date'].max() + pd.DateOffset(days=1),
                    periods=len(quantile_predictions),
                    freq='D'
                )
            elif config['freq_offset'] == 'M':
                future_dates = pd.date_range(
                    start=data[data['Entity'] == entity]['Date'].max() + pd.DateOffset(months=1),
                    periods=len(quantile_predictions),
                    freq='M'
                )
            elif config['freq_offset'] == 'Y':
                future_dates = pd.date_range(
                    start=data[data['Entity'] == entity]['Date'].max() + pd.DateOffset(years=1),
                    periods=len(quantile_predictions),
                    freq='Y'
                )
            elif config['freq_offset'] == 'Y1':
                future_dates = pd.date_range(
                    # start=data[data['Entity'] == entity]['Date'].max() + 1,
                    start=data[data['Entity'] == entity]['Date'].max() + pd.DateOffset(years=1),
                    periods=len(quantile_predictions),
                    freq='YS'
                )            
            # Create DataFrame with all quantiles
            entity_predictions = pd.DataFrame({
                'Entity': entity,
                'Date': future_dates,
                'Forecast_Step': range(1, len(quantile_predictions) + 1)
            })
            
            # Add quantile columns
            for i, quantile in enumerate(quantiles):
                if i < quantile_predictions.shape[1]:
                    col_name = f'Quantile_{quantile:.2f}'
                    entity_predictions[col_name] = quantile_predictions[:, i]
            
            # Add median as the main prediction (if available)
            median_idx = None
            for i, q in enumerate(quantiles):
                if abs(q - 0.5) < 0.01:  # Find quantile closest to 0.5
                    median_idx = i
                    break
            
            if median_idx is not None:
                entity_predictions['Predicted_Value'] = quantile_predictions[:, median_idx]
            else:
                # If no median quantile, use the middle quantile
                mid_idx = len(quantiles) // 2
                entity_predictions['Predicted_Value'] = quantile_predictions[:, mid_idx]
            
            predictions_list.append(entity_predictions)
            
        except Exception as e:
            print(f"Error forecasting for entity {entity}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all predictions
    if predictions_list:
        all_predictions = pd.concat(predictions_list, ignore_index=True)
        print('cols all ', all_predictions.columns)
        
        # Save predictions to CSV
        output_path = BASE_DIR / "app" / f"{dataset_key}_forecasts_30_steps_quantiles.csv"
        all_predictions.to_csv(output_path, index=False)
        print(f"Forecasts saved to: {output_path}")
        
        # Display summary
        print(f"\nForecast Summary for {config['dataset_name']}:")
        print(f"Total entities forecasted: {all_predictions['Entity'].nunique()}")
        print(f"Forecast horizon: 30 steps ({config['time_idx_type']})")
        print(f"Date range: {all_predictions['Date'].min()} to {all_predictions['Date'].max()}")
        
        # Show available quantiles
        quantile_cols = [col for col in all_predictions.columns if col.startswith('Quantile_')]
        print(f"Available quantiles: {quantile_cols}")
        
        # Display sample predictions
        print("\nSample predictions:")
        for entity in all_predictions['Entity'].unique()[:3]:  # Show first 3 entities
            entity_preds = all_predictions[all_predictions['Entity'] == entity]
            display_cols = ['Date', 'Predicted_Value', 'Forecast_Step'] + quantile_cols[:3]  # Show first 3 quantiles
            print(f"\n{entity}:")
            print(entity_preds[display_cols].head())
        
        # Create visualization with quantiles
        create_forecast_plots_with_quantiles(data, all_predictions, config)
        
        return all_predictions
    else:
        print("No successful forecasts generated.")
        return None


def create_forecast_plots_with_quantiles(historical_data, predictions, config):
    """Create visualization of forecasts with quantile bands"""
    print("Creating forecast plots with quantiles...")
    
    # Get quantile columns
    quantile_cols = [col for col in predictions.columns if col.startswith('Quantile_')]
    print("Prediction columns")
    
    for col in quantile_cols:
        print(" ", col)
    quantile_values = []
    for col in quantile_cols:
        try:
            q_val = float(col.split('_')[1])
            quantile_values.append(q_val)
        except:
            continue
    
    # Sort quantiles
    quantile_data = list(zip(quantile_values, quantile_cols))
    quantile_data.sort()
    
    # Plot for each entity (limit to first 6 entities for readability)
    entities_to_plot = predictions['Entity'].unique()[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.flatten()
    
    for i, entity in enumerate(entities_to_plot):
        if i >= 6:
            break
            
        ax = axes[i]
        
        # Historical data
        hist_data = historical_data[historical_data['Entity'] == entity]
        hist_data = hist_data.sort_values('Date')
        
        # Predictions for this entity
        entity_preds = predictions[predictions['Entity'] == entity].sort_values('Date')
        
        # Plot historical data
        ax.plot(hist_data['Date'], hist_data['Value'], 
                label='Historical', color='blue', alpha=0.7, linewidth=2)
        
        # Plot median prediction
        ax.plot(entity_preds['Date'], entity_preds['Predicted_Value'], 
                label='Forecast (Median)', color='red', marker='o', markersize=4, linewidth=2)
        
        # Plot quantile bands optimized for your specific quantiles
        if len(quantile_data) >= 2:
            # Define specific quantile pairs for uncertainty bands
            quantile_pairs = [
                (0.02, 0.98, 'lightcoral', 0.15, '2%-98% (Very Wide)'),
                (0.1, 0.9, 'lightsalmon', 0.25, '10%-90% (Wide)'),
                (0.25, 0.75, 'mistyrose', 0.35, '25%-75% (IQR)')
            ]
            
            # Create uncertainty bands
            for lower_q, upper_q, color, alpha, label in quantile_pairs:
                lower_col = f'Quantile_{lower_q:.2f}'
                upper_col = f'Quantile_{upper_q:.2f}'
                
                if lower_col in entity_preds.columns and upper_col in entity_preds.columns:
                    ax.fill_between(
                        entity_preds['Date'],
                        entity_preds[lower_col],
                        entity_preds[upper_col],
                        alpha=alpha,
                        color=color,
                        label=label
                    )
        
        # Plot key quantile lines (excluding median and extreme quantiles from bands)
        key_quantiles = [(0.02, 'dotted', 'darkred', 'Q2%'), 
                        (0.1, 'dashed', 'red', 'Q10%'),
                        (0.9, 'dashed', 'red', 'Q90%'),
                        (0.98, 'dotted', 'darkred', 'Q98%')]
        
        for q_val, linestyle, color, label in key_quantiles:
            q_col = f'Quantile_{q_val:.2f}'
            if q_col in entity_preds.columns:
                ax.plot(entity_preds['Date'], entity_preds[q_col], 
                       linestyle=linestyle, alpha=0.7, linewidth=1.5,
                       color=color, label=label)
        
        # Add vertical line to separate historical and forecast
        if len(hist_data) > 0:
            ax.axvline(x=hist_data['Date'].max(), color='gray', 
                      linestyle='--', alpha=0.5, label='Forecast Start')
        
        ax.set_title(f'{config["dataset_name"]} Forecast - {entity}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        
        # Limit legend items to avoid clutter - prioritize most important items
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 10:  # Show only most important items
            # Priority order: Historical, Median, Bands, Key quantiles
            priority_keywords = ['historical', 'median', 'wide', 'iqr', 'forecast start']
            important_indices = []
            
            for keyword in priority_keywords:
                for idx, label in enumerate(labels):
                    if keyword.lower() in label.lower() and idx not in important_indices:
                        important_indices.append(idx)
                        if len(important_indices) >= 8:
                            break
                if len(important_indices) >= 8:
                    break
            
            # Add remaining items if we have space
            for idx in range(len(labels)):
                if idx not in important_indices and len(important_indices) < 10:
                    important_indices.append(idx)
            
            handles = [handles[i] for i in important_indices[:10]]
            labels = [labels[i] for i in important_indices[:10]]
        
        ax.legend(handles, labels, fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Remove empty subplots
    for i in range(len(entities_to_plot), 6):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    # Save plot
    plot_path = BASE_DIR / "app" / "forecasts" / f"{config['dataset_name'].lower().replace(' ', '_')}_forecasts_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Forecast plot saved to: {plot_path}")
    plt.show()


def create_individual_forecast_plot(entity_data, predictions_data, config, entity_name):
    """Create a detailed plot for a single entity with all quantiles"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Get quantile columns
    quantile_cols = [col for col in predictions_data.columns if col.startswith('Quantile_')]
    
    # Historical data
    hist_data = entity_data.sort_values('Date')
    
    # Predictions for this entity
    entity_preds = predictions_data.sort_values('Date')
    
    # Plot historical data
    ax.plot(hist_data['Date'], hist_data['Value'], 
            label='Historical', color='blue', alpha=0.8, linewidth=2)
    
    # Plot median prediction
    ax.plot(entity_preds['Date'], entity_preds['Predicted_Value'], 
            label='Forecast (Median)', color='red', marker='o', markersize=3, linewidth=2)
    
    # Plot all quantiles with specific styling for your quantiles
    quantile_styles = {
        0.02: ('dotted', 'darkred', 0.6),
        0.1: ('dashed', 'red', 0.7),
        0.25: ('dashdot', 'orange', 0.7),
        0.5: ('solid', 'black', 0.9),  # Median - will be plotted separately
        0.75: ('dashdot', 'green', 0.7),
        0.9: ('dashed', 'blue', 0.7),
        0.98: ('dotted', 'darkblue', 0.6)
    }
    
    for i, col in enumerate(quantile_cols):
        q_val = float(col.split('_')[1])
        if q_val in quantile_styles and q_val != 0.5:  # Don't plot median again
            linestyle, color, alpha = quantile_styles[q_val]
            ax.plot(entity_preds['Date'], entity_preds[col], 
                   color=color, alpha=alpha, linestyle=linestyle, linewidth=1.5,
                   label=f'Q{q_val:.0%}' if q_val >= 0.1 else f'Q{q_val:.1%}')
    
    # Add vertical line
    if len(hist_data) > 0:
        ax.axvline(x=hist_data['Date'].max(), color='gray', 
                  linestyle='--', alpha=0.5, label='Forecast Start')
    
    ax.set_title(f'Detailed Forecast - {entity_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig


def print_quantile_ranges(predictions):
    """Print quantile ranges for each entity"""
    print("\n" + "="*60)
    print("QUANTILE FORECAST RANGES")
    print("="*60)
    
    quantile_cols = [col for col in predictions.columns if col.startswith('Quantile_')]
    
    for entity in predictions['Entity'].unique()[:5]:  # Show first 5 entities
        print(f"\n{entity}:")
        print("-" * 40)
        entity_preds = predictions[predictions['Entity'] == entity]
        
        # Calculate ranges for key forecast steps
        key_steps = [1, 15, 30]  # First, middle, and last forecast
        
        for step in key_steps:
            if step <= len(entity_preds):
                step_data = entity_preds[entity_preds['Forecast_Step'] == step].iloc[0]
                print(f"  Step {step:2d}: ", end="")
                
                # Show key quantiles
                key_quantiles = [0.02, 0.25, 0.5, 0.75, 0.98]
                values = []
                for q in key_quantiles:
                    col = f'Quantile_{q:.2f}'
                    if col in step_data:
                        values.append(f"{step_data[col]:.2f}")
                
                print(" | ".join([f"Q{int(q*100):2d}%: {v:>7s}" for q, v in zip(key_quantiles, values)]))
        
        # Show uncertainty width (98th - 2nd percentile)
        if 'Quantile_0.02' in entity_preds.columns and 'Quantile_0.98' in entity_preds.columns:
            uncertainty_width = (entity_preds['Quantile_0.98'] - entity_preds['Quantile_0.02']).mean()
            print(f"  Avg Uncertainty Width (98%-2%): {uncertainty_width:.2f}")


def save_quantile_analysis(predictions, config):
    """Save detailed quantile analysis"""
    analysis_data = []
    
    quantile_cols = [col for col in predictions.columns if col.startswith('Quantile_')]
    
    for entity in predictions['Entity'].unique():
        entity_preds = predictions[predictions['Entity'] == entity]
        
        # Calculate uncertainty metrics
        if 'Quantile_0.02' in entity_preds.columns and 'Quantile_0.98' in entity_preds.columns:
            uncertainty_96 = entity_preds['Quantile_0.98'] - entity_preds['Quantile_0.02']
        else:
            uncertainty_96 = None
            
        if 'Quantile_0.25' in entity_preds.columns and 'Quantile_0.75' in entity_preds.columns:
            iqr = entity_preds['Quantile_0.75'] - entity_preds['Quantile_0.25']
        else:
            iqr = None
        
        # Add analysis row
        analysis_row = {
            'Entity': entity,
            'Forecast_Horizon': len(entity_preds),
            'Median_Forecast_Mean': entity_preds['Predicted_Value'].mean(),
            'Median_Forecast_Std': entity_preds['Predicted_Value'].std(),
        }
        
        # Add individual quantile statistics
        for col in quantile_cols:
            q_val = float(col.split('_')[1])
            analysis_row[f'Q{q_val:.0%}_Mean' if q_val >= 0.1 else f'Q{q_val:.1%}_Mean'] = entity_preds[col].mean()
        
        # Add uncertainty metrics
        if uncertainty_96 is not None:
            analysis_row['Uncertainty_96pct_Mean'] = uncertainty_96.mean()
            analysis_row['Uncertainty_96pct_Max'] = uncertainty_96.max()
        
        if iqr is not None:
            analysis_row['IQR_Mean'] = iqr.mean()
            analysis_row['IQR_Max'] = iqr.max()
        
        analysis_data.append(analysis_row)
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # Save analysis
    analysis_path = BASE_DIR / "app" / f"{config['dataset_name'].lower().replace(' ', '_')}_quantile_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)
    print(f"Quantile analysis saved to: {analysis_path}")
    
    return analysis_df


def list_available_datasets():
    """List all available datasets"""
    print("Available datasets:")
    for key, config in DATASET_CONFIGS.items():
        print(f"  - {key}: {config['dataset_name']} ({config['time_idx_type']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Forecast time series data using trained TFT models with quantile predictions')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=list(DATASET_CONFIGS.keys()),
                       help='Dataset to forecast')
    parser.add_argument('--list', action='store_true', 
                       help='List available datasets')
    parser.add_argument('--entity', type=str, 
                       help='Create detailed plot for specific entity')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
    else:
        try:
            # forecasts = forecast_next_30_steps(args.dataset)
            forecasts = forecast_next_30_steps(args.dataset)
            if forecasts is not None:
                print(f"\nForecasting completed successfully for {args.dataset}!")
                
                # Print quantile ranges
                print_quantile_ranges(forecasts)
                
                # Save quantile analysis
                config = DATASET_CONFIGS[args.dataset]
                save_quantile_analysis(forecasts, config)
                
                # Create detailed plot for specific entity if requested
                if args.entity and args.entity in forecasts['Entity'].values:
                    entity_preds = forecasts[forecasts['Entity'] == args.entity]
                    # You would need historical data for this entity
                    hist_data=args.dataset.sort_values('Date').tail(30)
                    entity_hist = hist_data[hist_data['Entity'] == args.entity]
                    fig = create_individual_forecast_plot(entity_hist, entity_preds, config, args.entity)
                    plt.show()
                    print(f"Individual plot for {args.entity} would be created here")
                
            else:
                print(f"\nForecasting failed for {args.dataset}!")
        except Exception as e:
            print(f"Error during forecasting: {str(e)}")
            import traceback
            traceback.print_exc()

