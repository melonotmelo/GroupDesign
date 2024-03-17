import einops
import torch
import torch.nn as nn
import math
from timm.models.layers import Mish, Swish
from utils.rope import (
    precompute_freqs_cis_1d,
    precompute_freqs_cis_2d,
    apply_rotary_emb
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Activation(nn.Module):
    def __init__(self, activation=None):
        super().__init__()
        if activation == 'mish':
            self.activation = Mish()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = Swish()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = config['hidden_dim']
        num_heads = config['num_heads']
        attn_dropout = config['dropout_rate']
        self.L3d_train = (config['train_img_size'] // config['patch_size']) ** 3
        self.L2d_train = (config['train_img_size'] // config['patch_size']) ** 2
        self.L1d_train = (config['train_img_size'] // config['patch_size'])

        self.L3d_eval = (config['eval_img_size'] // config['patch_size']) ** 3
        self.L2d_eval = (config['eval_img_size'] // config['patch_size']) ** 2
        self.L1d_eval = (config['eval_img_size'] // config['patch_size'])

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} should be divisible by num_heads {num_heads}"
            )
        self.num_t = config['num_t']
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        self._init_rope()

        self.W = torch.nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.activation = Activation("swish")
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.out_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

    def _init_rope(self):
        base = 10000.0
        factor = 1.0

        self.freqs_cis_spatial_1d_train = precompute_freqs_cis_1d(
            self.head_dim, self.L1d_train, x_factor=1.0)
        self.freqs_cis_spatial_2d_train = precompute_freqs_cis_2d(
            self.head_dim, self.L1d_train, self.L1d_train, x_factor=1.0, y_factor=1.0)
        self.freqs_cis_spatial_1d_eval = precompute_freqs_cis_1d(
            self.head_dim, self.L1d_eval, x_factor=factor, base=base)
        self.freqs_cis_spatial_2d_eval = precompute_freqs_cis_2d(
            self.head_dim, self.L1d_eval, self.L1d_eval, x_factor=factor, y_factor=factor, base=base)
        self.freqs_cis_temporal = precompute_freqs_cis_1d(self.head_dim, self.num_t)

    @staticmethod
    def _make_causal_mask(
            inputs_ids_shape: torch.Size,
            dtype: torch.dtype,
            device: torch.device,
    ):
        batch_size, seq_len = inputs_ids_shape
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < ((mask_cond + 1).view(mask.size(-1), 1)), 0)

        mask = mask.to(dtype)

        return mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)

    @staticmethod
    def _expand_mask(
            mask: torch.Tensor,
            dtype: torch.dtype,
            batch_size: int,
    ):

        """
            expand mask [seq_len, seq_len] to [bsz, 1, seq_len, seq_len]
        """
        src_len, tgt_len = mask.size()
        expanded_mask = mask[None, None, :, :].expand(batch_size, 1, src_len, tgt_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    def forward(self, x, attention_mask=None):
        # shape of x: [batch_size,len,hidden_dim]
        # shape of attn_mask: [len,len]
        batch_size, length, _ = x.size() # length = num_t * patch_len

        # Q,K,V: [batch_size*num_heads,len, hidden_dim//num_heads]
        q, k, v = einops.rearrange(
            self.W(x), "b l (n h)->(b n) l h", n=self.num_heads).chunk(3, dim=-1)
        # RoPE on spatial
        if self.training:
            freqs_cis_spatial = self.freqs_cis_spatial_1d_train.to(x.device) \
                if length == self.L1d_train * self.num_t \
                else self.freqs_cis_spatial_2d_train.to(x.device) \
                if length == self.L1d_train * self.L1d_train * self.num_t else None
        else: # evaluate mode
            freqs_cis_spatial = self.freqs_cis_spatial_1d_eval.to(x.device) \
                if length == self.L1d_eval * self.num_t \
                else self.freqs_cis_spatial_2d_eval.to(x.device) \
                if length == self.L1d_eval * self.L1d_eval * self.num_t else None

        if freqs_cis_spatial is None:
            raise ValueError(f"length {length} is not equal to patch_size * num_t!")
        # apply rotary matrix on spatial dimension
        q = einops.rearrange(q, "b (t l) h->b t l h", t=self.num_t)
        k = einops.rearrange(k, "b (t l) h->b t l h", t=self.num_t)
        q, k = apply_rotary_emb(q, k, freqs_cis_spatial)

        # RoPE on temporal
        q = einops.rearrange(q, "b t l h->b l t h", t=self.num_t)
        k = einops.rearrange(k, "b t l h->b l t h", t=self.num_t)
        q, k = apply_rotary_emb(q, k, self.freqs_cis_temporal.to(q.device))
        q = einops.rearrange(q, "(b n) l t h->b (t l) n h", n=self.num_heads, t=self.num_t)
        k = einops.rearrange(k, "(b n) l t h->b (t l) n h", n=self.num_heads, t=self.num_t)
        v = einops.rearrange(v, "(b n) l h->b l n h", n=self.num_heads)

        assert q.shape[-2] == self.num_heads and k.shape[-2] == self.num_heads and v.shape[-2] == self.num_heads
        q = einops.rearrange(q, "b l n h->b n l h", n=self.num_heads)
        k = einops.rearrange(k, "b l n h->b n l h", n=self.num_heads)
        v = einops.rearrange(v, "b l n h->b n l h", n=self.num_heads)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        expanded_attention_mask = self._make_causal_mask(
            (batch_size, length), attn_weights.dtype, device=attn_weights.device)
        if attention_mask is not None:
            attention_mask = self._expand_mask(attention_mask, attn_weights.dtype, batch_size)\
                .to(attn_weights.device)
            expanded_attention_mask = attention_mask.masked_fill(
                expanded_attention_mask.bool(), torch.finfo(expanded_attention_mask.dtype).min)

        attn_weights = attn_weights + expanded_attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        if attn_output.size() != (batch_size, self.num_heads, length, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, length, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = einops.rearrange(attn_output, "b l n h->b l (n h)", b=batch_size, n=self.num_heads)
        attn_output = self.activation(attn_output)
        attn_output = self.out_linear(attn_output)
        return attn_output


class EncoderLayer(torch.nn.Module):
    def __init__(self, **config):
        super().__init__()
        hidden_dim = config['hidden_dim']
        num_heads = config['num_heads']
        dropout_rate = config['dropout_rate']

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(config)
        self.norm1 = torch.nn.LayerNorm(self.hidden_dim)
        self.norm2 = torch.nn.LayerNorm(self.hidden_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            Activation("swish"),
            torch.nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask=None):
        x_copy = x.clone()
        x = self.attention(x, attention_mask)
        x = self.dropout(x) + x_copy
        x = self.norm1(x)
        x = self.feed_forward(x) + x
        x = self.norm2(x)
        return x


class Transformer(torch.nn.Module):
    """
    config includes:
    hidden_dim: int
    num_heads: int
    dropout_rate: float
    num_hidden_layers: int
    """

    def __init__(self, **config):
        super().__init__()
        num_hidden_layers = config['num_hidden_layers'] if 'num_hidden_layers' in config else 12
        self.encoders = torch.nn.ModuleList([EncoderLayer(**config) for _ in range(num_hidden_layers)])

    def forward(self, x, attention_mask):
        for encoder in self.encoders:
            x = encoder(x, attention_mask)
        return x


def test_transformer():
    transformer = Transformer(
        num_hidden_layers=12,
        hidden_dim=128,
        num_heads=8,
        dropout_rate=0.1,
        max_seq_len=64,
        img_size=64,
        patch_size=16,
        num_t=10,
    )

    x = torch.randn(2, 640, 128)
    transformer(x)
    x = torch.randn(2, 160, 128)
    transformer(x)
    x = torch.randn(2, 40, 128)
    transformer(x)
