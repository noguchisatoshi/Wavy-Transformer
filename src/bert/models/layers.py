# models/layers.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import CustomBertSelfAttention, JointLayerNorm

class CustomBertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        return self.intermediate_act_fn(self.dense(hidden_states))

class CustomBertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, dropout_prob=0.1):
        super().__init__()

        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states, residual_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return self.LayerNorm(hidden_states + residual_tensor)

class JointLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x, mode='pos'):
        if mode == 'pos':
            return F.linear(x, self.weight, self.bias)
        elif mode == 'vel':
            return F.linear(x, self.weight, None)
        else:
            raise ValueError("mode must be 'pos' or 'vel'")

class CustomBertIntermediateJoint(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation='gelu'):

        super(CustomBertIntermediateJoint, self).__init__()
        self.dense = JointLinear(hidden_size, intermediate_size, bias=True)

        if activation.lower() == 'gelu':
            self.intermediate_act_fn = nn.GELU()
            self.activation_derivative = self.gelu_derivative
        elif activation.lower() == 'relu':
            self.intermediate_act_fn = nn.ReLU()
            self.activation_derivative = self.relu_derivative
        else:
            raise ValueError("activation must be 'gelu' or 'relu'")

    def gelu_derivative(self, x):
        """
        Derivative of GLUE
        GELU(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
        GELU'(x) = 0.5 * (1 + erf(x/sqrt(2))) + (x / sqrt(2π)) * exp(-x^2/2)
        """
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) + (x * torch.exp(-0.5 * x ** 2)) / math.sqrt(2 * math.pi)

    def relu_derivative(self, x):
        """
        Derivative of RELU
        ReLU(x) = max(0, x) 
        ReLU'(x) = 1 (in case of x > 0), 0 (otherwise)
        """
        return (x > 0).type(x.dtype)

    def forward(self, pos, vel):
        pos_linear = self.dense(pos, mode='pos')
        vel_linear = self.dense(vel, mode='vel')
        
        pos_out = self.intermediate_act_fn(pos_linear)

        act_grad = self.activation_derivative(pos_linear)
        vel_out = act_grad * vel_linear
        
        return pos_out, vel_out

class CustomBertOutputJoint(nn.Module):
    def __init__(self, intermediate_size, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.dense = JointLinear(intermediate_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.joint_layer_norm = JointLayerNorm(hidden_size, eps=1e-12)

    def forward(self, pos, pos_residual, vel, vel_residual):
        pos_out = self.dense(pos, mode='pos')
        pos_out = self.dropout(pos_out)
        pos_out = pos_out + pos_residual

        vel_out = self.dense(vel, mode='vel')
        vel_out = self.dropout(vel_out)
        vel_out = vel_out + vel_residual

        pos_out, vel_out = self.joint_layer_norm(pos_out, vel_out)
        return pos_out, vel_out

class CustomBertLayer(nn.Module):
    def __init__(self,
                 hidden_size=768,
                 intermediate_size=3072,
                 num_attention_heads=12,
                 dropout_prob=0.1,
                 residual_type="diffuse",   # "diffuse", "wave", "mix"
                 tau=1.0,
                 apply_velocity_transform=False,
                 ):
        super().__init__()
        self.residual_type = residual_type
        self.add_mass = add_mass
        self.apply_velocity_transform = apply_velocity_transform

        self.attention = CustomBertSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout_prob=dropout_prob,
            residual_type=residual_type,
            tau=tau,
            )

        if self.residual_type in ["wave", "mix"] and self.apply_velocity_transform:
            self.intermediate_joint = CustomBertIntermediateJoint(hidden_size, intermediate_size)
            self.output_joint = CustomBertOutputJoint(intermediate_size, hidden_size, dropout_prob)

        else:
            self.intermediate = CustomBertIntermediate(hidden_size, intermediate_size)
            self.output = CustomBertOutput(intermediate_size, hidden_size, dropout_prob)

    def forward(self, hidden_states, attention_mask=None, previous_hidden_states=None, previous_hidden_velocity=None):
        if self.residual_type in ["wave", "mix"]:
            if previous_hidden_velocity is None:
                raise ValueError("previous_hidden_velocity is required for symplectic wave or mix residual in Self-Attention")
            
            attention_output, velocity, attn_weights = self.attention(
                hidden_states, 
                attention_mask=attention_mask, 
                previous_hidden_velocity=previous_hidden_velocity
            )

            if self.apply_velocity_transform:
                intermediate_output, intermediate_velocity = self.intermediate_joint(attention_output, velocity)
                output_states, output_velocity = self.output_joint(intermediate_output, attention_output, intermediate_velocity, velocity)
            else:
                intermediate_output = self.intermediate(attention_output)
                output_states = self.output(intermediate_output, attention_output)
                output_velocity = velocity
            
            return output_states, output_velocity, attn_weights

        else:
            attention_output, attn_weights = self.attention(hidden_states, attention_mask=attention_mask)
            intermediate_output = self.intermediate(attention_output)
            output_states = self.output(intermediate_output, attention_output)
            return output_states, attn_weights