import torch
import torch.nn as nn
from timm.models.layers import Mlp

class JointMlp(Mlp):
    """
    JointMlp:
    Based on timm’s Mlp, this module performs the two linear
    transformations (fc1, fc2) jointly for position (x) and
    velocity (v) so we can reduce CUDA kernel launches.

    • Position branch: applies a bias-included transform.  
    • Velocity branch: applies a bias-free transform and is
      scaled by the derivative of the activation function
      evaluated at the position branch’s pre-activation (Wx + b).

    Inputs and outputs are tuples (x, v).
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        # Call timm’s Mlp initializer
        super(JointMlp, self).__init__(
            in_features, hidden_features, out_features, act_layer, drop
        )
        # Choose the correct activation derivative
        if isinstance(self.act, nn.ReLU):
            self.act_deriv = lambda x: (x > 0).float()
        elif isinstance(self.act, nn.GELU):
            self.act_deriv = self._gelu_deriv
        else:
            # Define a derivative for other activations if necessary
            self.act_deriv = lambda x: torch.ones_like(x)

    def _gelu_deriv(self, x):
        # Approximate derivative of GELU:
        # d/dx GELU(x) = 0.5 (1 + erf(x/√2)) + x/√(2π) · exp(−x²/2)
        sqrt_2   = torch.sqrt(torch.tensor(2.0, device=x.device))
        sqrt_2pi = torch.sqrt(torch.tensor(2 * 3.141592653589793, device=x.device))
        return 0.5 * (1.0 + torch.erf(x / sqrt_2)) + (x / sqrt_2pi) * torch.exp(-0.5 * x**2)

    def forward(self, x, v):
        """
        x, v: both shapes [B, in_features]  
        Returns (x_out, v_out)
        """
        B = x.shape[0]

        # fc1 — position branch with bias, velocity branch without bias
        x_fc1 = torch.matmul(x, self.fc1.weight.t()) + self.fc1.bias
        v_fc1 = torch.matmul(v, self.fc1.weight.t())

        # Position branch: apply activation
        x_act = self.act(x_fc1)

        # Velocity branch: scale by activation derivative of position branch
        v_scale = self.act_deriv(x_fc1)
        v_act   = v_fc1 * v_scale

        # Concatenate the two branches and apply dropout
        cat_act = torch.cat([x_act, v_act], dim=0)
        cat_act = self.drop(cat_act)

        # fc2 — linear transform of the concatenated tensor
        out_fc2 = torch.matmul(cat_act, self.fc2.weight.t())
        # Apply bias to the x branch only
        out_fc2 = torch.cat([out_fc2[:B] + self.fc2.bias, out_fc2[B:]], dim=0)
        out_fc2 = self.drop(out_fc2)

        # Split and return
        x_out, v_out = out_fc2[:B], out_fc2[B:]
        return x_out, v_out


class JointLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta  = nn.Parameter(torch.zeros(normalized_shape))
        self.eps   = eps

    def forward(self, x, v):
        """
        x: [batch_size, seq_len, hidden_size] — position  
        v: [batch_size, seq_len, hidden_size] — velocity
        """
        mean    = x.mean(dim=-1, keepdim=True)
        var     = x.var(dim=-1, keepdim=True, unbiased=False)
        inv_std = 1.0 / torch.sqrt(var + self.eps)

        # Normalize with the same statistics
        x_norm = (x - mean) * inv_std
        v_norm = v * inv_std

        # Shared scale and shift
        x_norm = x_norm * self.gamma + self.beta
        v_norm = v_norm * self.gamma
        return x_norm, v_norm