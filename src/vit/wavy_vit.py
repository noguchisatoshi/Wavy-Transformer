############### Wavy Transformer based ViT ###############

import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model

from vit import Attention, _init_vit_weights, _load_weights
from utils_wavy import JointLayerNorm, JointMlp

__all__ = [
'small_12_wave', 'featscale_small_12_wave', 'tiny_12_wave', 'featscale_tiny_12_wave'
]

class WavyBlockParallel(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_wave=JointLayerNorm):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_wave(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.lamb1 = nn.Parameter(torch.zeros(dim), requires_grad=True) 
        self.lamb2 = nn.Parameter(torch.zeros(dim), requires_grad=True)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_wave(dim)
  
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = JointMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.beta = nn.Parameter(torch.full((dim,), -8.0), requires_grad=True)

    def freq_decompose(self, x):
        x_d = torch.mean(x, -2, keepdim=True) # [bs, 1, dim]
        x_h = x - x_d # high freq [bs, len, dim]
        return x_d, x_h
    
    def featscale(self, x, lamb1, lamb2):
        x_d, x_h = self.freq_decompose(x)
        x_d = x_d * (1 + lamb1)
        x_h = x_h * (1 + lamb2)
        x_output = x_d + x_h
        return x_output

    def forward(self, x, v):
        B, N, D = x.shape
        mix_alpha = self.beta.sigmoid()

        # wave
        x_norm, v_norm = self.norm1(x, v)
        x_attn = self.attn(x_norm)

        x_attn = self.featscale(x_attn, self.lamb1, self.lamb2)
        attn_output = self.drop_path(x_attn)

        delta = attn_output - x
        
        v          = 0.5 * delta + v
        x_wave     = 0.5 * v + x
        x_diffuse  = x + attn_output

        x = mix_alpha * x_wave + (1 - mix_alpha) * x_diffuse

        x_tmp, v_tmp = self.norm2(x, v)
        x_tmp, v_tmp = self.mlp(x_tmp, v_tmp)

        x = x + self.drop_path(x_tmp)
        v = v + self.drop_path(v_tmp)

        return x, v

    def flops(self, N):
        f = 2 * self.norm1.gamma.numel() * N  # approx norm
        f += self.attn.flops(N)
        f += 2 * self.lamb1.numel() * N
        f += 4 * N * self.lamb1.numel() * (self.lamb1.numel() * (self.lamb1.numel() / self.lamb1.numel()))
        f += 2 * self.norm2.gamma.numel() * N
        f += 4 * self.lamb1.numel() * N
        return f

class WavyBlockParallelWithoutFeatScale(WavyBlockParallel):
    """Same as WavyBlockParallel but without feature scaling."""
    def forward(self, x, v):
        x, v = self.norm1(x, v)
        out1 = self.drop_path(self.attn(x))
        delta = out1 - x
        v = 0.5 * delta + v
        x_wave = 0.5 * v + x
        x = self.beta.sigmoid() * x_wave + (1 - self.beta.sigmoid()) * (x + out1)

        x2, v2 = self.norm2(x, v)
        dx, dv = self.mlp(x2, v2)
        return x + self.drop_path(dx), v + self.drop_path(dv)


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', residual_type="diffuse"):

        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): pre-logits size
            distilled (bool): include distillation token & head
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: normalization layer
            act_layer: activation layer
            weight_init (str): weight initialization scheme
            residual_type (str): one of "diffuse", "wave", etc.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.residual_type = residual_type

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if self.residual_type == "diffuse":
            self.blocks = nn.Sequential(*[
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
                for i in range(depth)])
            self.norm = norm_layer(embed_dim)

        elif self.residual_type == "wave":
            self.blocks = nn.Sequential(*[
                WavyBlockParallel(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_wave=JointLayerNorm, act_layer=act_layer)
                for i in range(depth)])
            self.norm = norm_layer(embed_dim)
        
        elif self.residual_type == "wave_without_featscale":
            self.blocks = nn.Sequential(*[
                WavyBlockParallelWithoutFeatScale(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_wave=JointLayerNorm, act_layer=act_layer)
                for i in range(depth)])
            self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.residual_type == "diffuse":
            x = self.blocks(x)
            x = self.norm(x)

        elif self.residual_type == "wave" or self.residual_type == "wave_without_featscale":
            v = torch.zeros(x.size(), device=x.device, dtype=x.dtype)
            for block in self.blocks:
                x, v = block(x, v)
            x, _ = self.norm(x, v)

        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

    def flops(self):
        # patch embed
        Ho = Wo = self.img_size // self.patch_size
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size * self.patch_size)
        # if self.norm is not None:
        #     flops += Ho * Wo * self.embed_dim

        # attn blocks
        for i, layer in enumerate(self.blocks):
            flops += layer.flops(Ho * Wo)
        flops += Ho * Wo * self.embed_dim

        # mlp readout
        flops += self.num_features * self.num_classes
        return flops


@register_model
def featscale_small_12_wave(pretrained=False, **kwargs):
    print("Running FeatScale small 12 based on wave residual!!")
    model = VisionTransformer(
        img_size=224, patch_size=16,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(JointLayerNorm, eps=1e-6), residual_type="wave", **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError()
    return model

@register_model
def featscale_tiny_12_wave(pretrained=False, **kwargs):
    print("Running FeatScale tiny 12 based on wave residual!!")
    model = VisionTransformer(
        img_size=224, patch_size=16,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(JointLayerNorm, eps=1e-6), residual_type="wave", **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError()
    return model

@register_model
def tiny_12_wave(pretrained=False, **kwargs):
    print("Running tiny 12 based on wave residual withtout any of featscale and attnscale!!")
    model = VisionTransformer(
        img_size=224, patch_size=16,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(JointLayerNorm, eps=1e-6), residual_type="wave_without_featscale", **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError()
    return model

@register_model
def small_12_wave(pretrained=False, **kwargs):
    print("Running tiny 12 based on wave residual withtout any of featscale and attnscale!!")
    model = VisionTransformer(
        img_size=224, patch_size=16,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(JointLayerNorm, eps=1e-6), residual_type="wave_without_featscale", **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError()
    return model