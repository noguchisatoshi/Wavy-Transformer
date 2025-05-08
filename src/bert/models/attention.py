# models/attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class JointLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta  = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x, v):
        """
        x: [batch_size, seq_len, hidden_size] -- position
        v: [batch_size, seq_len, hidden_size] -- velocity
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        v_norm = v / torch.sqrt(var + self.eps)
        x_norm = x_norm * self.gamma + self.beta
        v_norm = v_norm * self.gamma
        return x_norm, v_norm

class CustomBertSelfAttention(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_attention_heads: int, 
        dropout_prob: float = 0.1, 
        residual_type: str = "diffuse",
        tau: float = 1.0,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.residual_type = residual_type
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.tau = tau
        self.epsilon = 1e-8

        assert self.head_dim * self.num_attention_heads == self.hidden_size, "hidden_size must be divisible by num_attention_heads"

        self.self = nn.ModuleDict({
            'query': nn.Linear(self.hidden_size, self.hidden_size),
            'key': nn.Linear(self.hidden_size, self.hidden_size),
            'value': nn.Linear(self.hidden_size, self.hidden_size)
        })
        self.attn_dropout = nn.Dropout(dropout_prob)

        if self.residual_type in ["wave", "mix"]:
            self.output = nn.ModuleDict({
                'dense': nn.Linear(self.hidden_size, self.hidden_size),
                'dropout': nn.Dropout(dropout_prob),
                'LayerNorm': JointLayerNorm(self.hidden_size)
            })
        else:
            self.output = nn.ModuleDict({
                'dense': nn.Linear(self.hidden_size, self.hidden_size),
                'dropout': nn.Dropout(dropout_prob),
                'LayerNorm': nn.LayerNorm(self.hidden_size)
            })
        
        if self.residual_type in ["mix"]:
            self.beta = nn.Parameter(torch.tensor(0.0))

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None, previous_hidden_states: torch.Tensor = None, previous_hidden_velocity: torch.Tensor = None):
        batch_size, seq_len, _ = hidden_states.size()

        mixed_query_layer = self.self['query'](hidden_states)
        mixed_key_layer = self.self['key'](hidden_states)
        mixed_value_layer = self.self['value'](hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_scores.clamp_(-1000.0, 1000.0)
        attention_probs = F.softmax(attention_scores, dim=-1)

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output = self.output['dense'](context_layer)
        output = self.output['dropout'](output)
        
        # simplectic integrater
        if self.residual_type == "wave":
            if previous_hidden_velocity is None:
                raise ValueError("previous_hidden_velocity is required for wave residual in Self-Attention")

            hidden_velocity = self.tau * (output - hidden_states) + previous_hidden_velocity
            output = self.tau * hidden_velocity + hidden_states
            output, hidden_velocity = self.output['LayerNorm'](output, hidden_velocity)
            return output, hidden_velocity, attention_probs

        elif self.residual_type == "mix":
            if previous_hidden_velocity is None:
                raise ValueError("previous_hidden_velocity is required for mix residual in Self-Attention")
            
            mix_alpha = torch.sigmoid(self.beta)

            delta = output - hidden_states  # [B, S, D]
                
            hidden_velocity = (
                mix_alpha * (self.tau * delta + previous_hidden_velocity) +
                (1 - mix_alpha) * delta
            )

            output = self.tau * hidden_velocity + hidden_states
            output, hidden_velocity = self.output['LayerNorm'](output, hidden_velocity)
            return output, hidden_velocity, attention_probs

        elif self.residual_type == "diffuse":
            if self.tau == 1.0:
                output = output + hidden_states
            else:
                output = self.tau * output + hidden_states

        output = self.output['LayerNorm'](output)
        return output, attention_probs
