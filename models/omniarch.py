import math

import einops
import torch
from torch import nn

from models.revin import RevIN
from models.transformer import Transformer
from others.data_information import *
from random import choice


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.",
              stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


# extend mask before attention (seq_len) -> (num_heads, seq_len, seq_len)


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


class OmniArchEncoder(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.num_t = config['num_t']
        self.hidden_dim = config['hidden_dim']
        self.channels = config['channels']
        self.patch_size = config['patch_size']
        self.conv3d = nn.Conv3d(
            self.channels,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True
        )
        self.weight = nn.Parameter(self.conv3d.weight)
        self.bias = nn.Parameter(self.conv3d.bias)

    def forward(self, phy_fls):
        dim_physics = len(phy_fls.size())
        if dim_physics == 6:
            weight = self.weight
            phy_fls = einops.rearrange(phy_fls, 'b t (m x) (n y) (o z) c -> b t c m x n y o z',
                                       x=self.patch_size, y=self.patch_size, z=self.patch_size)
            embedding = einops.einsum(weight, phy_fls, 'h c x y z,b t c m x n y o z -> b t h m n o') \
                        + self.bias.reshape(1, 1, -1, 1, 1, 1)
        elif dim_physics == 5:
            weight = torch.mean(self.weight, dim=[2])
            phy_fls = einops.rearrange(phy_fls, 'b t (m x) (n y) c -> b t c m x n y',
                                       x=self.patch_size, y=self.patch_size)
            embedding = einops.einsum(weight, phy_fls, 'h c x y,b t c m x n y -> b t h m n') \
                        + self.bias.reshape(1, 1, -1, 1, 1)
        elif dim_physics == 4:
            weight = torch.mean(self.weight, dim=[2, 3])
            phy_fls = einops.rearrange(phy_fls, 'b t (m x) c -> b t c m x ', x=self.patch_size)
            embedding = einops.einsum(weight, phy_fls, 'h c x,b t c m x -> b t h m') \
                        + self.bias.reshape(1, 1, -1, 1)
        else:
            raise ValueError(f"dim_physics should be 4, 5 or 6 for 1,2 and 3D respectively, but got {dim_physics}")
        return einops.rearrange(embedding, 'b t h ... -> b (t ...) h')


class OmniArchDecoder(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.num_t = config['num_t']
        self.hidden_dim = config['hidden_dim']
        self.channels = config['channels']
        self.patch_size = config['patch_size']
        self.weight = nn.Parameter(torch.randn((self.hidden_dim, self.channels,
                                                self.patch_size, self.patch_size, self.patch_size)))
        # self.bias is deleted, just for successfully loading ckpt
        self.bias = nn.Parameter(torch.randn((1, 1, 32, 32, 32, self.channels)))

    # B (T X (Y) (Z)) H
    def forward(self, embedding):
        image_size = self.config['train_img_size'] if self.training else self.config['eval_img_size']
        patch_side = image_size // self.patch_size

        embedding = einops.rearrange(embedding, 'b (t l) h -> b t h l', t=self.num_t)
        length = embedding.size(3)
        if length == (patch_side * patch_side * patch_side):
            weight = self.weight
            embedding = einops.rearrange(embedding, 'b t h (m n o) -> b t h m n o'
                                         , m=patch_side, n=patch_side, o=patch_side)
            phy_fls = einops.einsum(weight, embedding, 'h c x y z, b t h m n o -> b t c m x n y o z')
            phy_fls = einops.rearrange(phy_fls, 'b t c m x n y o z -> b t (m x) (n y) (o z) c')
        elif length == (patch_side * patch_side):
            weight = torch.mean(self.weight, dim=[2])
            embedding = einops.rearrange(embedding, 'b t h (m n) -> b t h m n'
                                         , m=patch_side, n=patch_side)
            phy_fls = einops.einsum(weight, embedding, 'h c x y, b t h m n -> b t c m x n y')
            phy_fls = einops.rearrange(phy_fls, 'b t c m x n y -> b t (m x) (n y) c')
        elif length == patch_side:
            weight = torch.mean(self.weight, dim=[2, 3])
            embedding = einops.rearrange(embedding, 'b t h m -> b t h m', m=patch_side)
            phy_fls = einops.einsum(weight, embedding, 'h c x, b t h m -> b t c m x')
            phy_fls = einops.rearrange(phy_fls, 'b t c m x -> b t (m x) c')
        else:
            raise ValueError(f"L should be for 1,2 and 3D respectively, but got {length}")
        return phy_fls


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class OmniArch(nn.Module):
    @staticmethod
    def get_mask(num_t, patch_num, phy_pred_method='patch'):
        if phy_pred_method == 'step':
            row = torch.arange(0, num_t * patch_num, step=patch_num)
            row = einops.repeat(row, 'i->(i j)', j=patch_num)
            column = row.reshape(-1, 1)
            dr = (row <= column).to(device)
        elif phy_pred_method == 'patch':
            phy_len = num_t * patch_num
            dr = ~torch.triu(torch.ones((phy_len, phy_len), dtype=torch.bool), diagonal=1).to(device)
        return dr

    def __init__(self, **config):
        super(OmniArch, self).__init__()
        # model configs
        self.num_t = config['num_t']
        self.patch_size = config['patch_size']
        self.phy_pred_method = config['phy_pred_method']

        # revin
        channels = config['channels']
        # self.revin = RevIN(channels)

        # model layers
        self.encoder = OmniArchEncoder(**config)
        self.backbone = Transformer(**config)
        self.decoder = OmniArchDecoder(**config)
        # init params
        self.apply(_init_weights)

    def forward(self, phy_fls, attention_mask=None):
        # phy_fls: B T C X (Y) (Z)
        # phy_fls = einops.rearrange(phy_fls, 'b t c ... -> b t ... c')
        # phy_fls = self.revin(phy_fls, "norm")
        embedding = self.encoder(phy_fls)
        # embedding: B (T X (Y) (Z)) H
        hidden_states = self.backbone(
            x=embedding, attention_mask=attention_mask
        )
        pred = self.decoder(hidden_states)
        # pred: B T X (Y) (Z) C
        pred = self.revin(pred, "denorm")
        pred = einops.rearrange(pred, 'b t ... c -> b t c ...')
        return pred


def RMSE(y, pred):
    B = y.size(0)
    T = y.size(1)
    C = y.size(2)
    eps = 1e-4
    err_mean = torch.sqrt(torch.mean((y.reshape(B, T, C, -1) - pred.reshape(B, T, C, -1)) ** 2, dim=3))
    err_RMSE = torch.mean(err_mean, dim=0)
    nrm = torch.sqrt(torch.mean(y.reshape(B, T, C, -1) ** 2, dim=3)) + eps
    err_nRMSE = torch.mean(err_mean / nrm, dim=0)
    return torch.mean(err_RMSE), torch.mean(err_nRMSE)


def pass_batch(model, phy_fls, tp, attention_mask=None):
    phy_fls = phy_fls.to(device=device, dtype=torch.float32)
    tp_list = tp.tolist()
    tp_set = set(tp_list)
    loss_info = {}
    for x in tp_name.values():
        loss_info[f'{x}_rmse'] = 0
        loss_info[f'{x}_nrmse'] = 0
    loss_info['loss'] = 0
    for _tp in tp_set:
        _indices = tp == _tp
        _eq_name = tp_name[_tp]
        _channels = type_channel_index[_tp]
        label_physics = phy_fls[_indices, 1:, ...]
        pred_physics = model(phy_fls[_indices, ...], attention_mask)[:, :-1, ...]
        _label = label_physics[:, :, _channels, ...]
        _pred = pred_physics[:, :, _channels, ...]
        print(f"calculate rmse & nrmse loss: label: {_label.shape}, pred: {_pred.shape}")
        # print(_label[0, 2, 0, 10:20], _pred[0, 2, 0, 10:20])
        _rmse, _nrmse = RMSE(_label, _pred)
        loss_info[f'{_eq_name}_rmse'] = _rmse
        loss_info[f'{_eq_name}_nrmse'] = _nrmse
        loss_info['loss'] += _nrmse
    return loss_info


if __name__ == '__main__':
    model_ = OmniArch(
        hidden_dim=128,
        patch_size=16,
        num_hidden_layers=12,
        num_t=10,
        train_img_size=64,
        eval_img_size=64,
        channels=4,
        phy_pred_method='step',
        num_heads=8,
        dropout_rate=0.1
    ).to(device)

    phy_fls_1d = torch.randn((1, 10, 4, 64)).to(device)
    phy_fls_2d = phy_fls_1d.unsqueeze(3).repeat(1, 1, 1, 64, 1)
    phy_fls_3d = phy_fls_2d.unsqueeze(3).repeat(1, 1, 1, 64, 1, 1)
    data = [phy_fls_1d, phy_fls_2d, phy_fls_3d]
    model_.eval()
    import random
    phy_fls = data[random.randint(0, 2)]
    with torch.no_grad():  # 不计算梯度，节省计算资源
        pred = model_(phy_fls)

    # 处理推理结果，例如打印输出
    print(pred)
    # optm = torch.optim.Adam(model_.parameters(), lr=0.0001)
    # import random
    # for e in range(10000):
    #     phy_fls = data[random.randint(0, 2)]
    #     label = phy_fls[:, :-1, ...]
    #     pred = model_(phy_fls)[:, 1:, ...]
    #     optm.zero_grad()
    #     loss = torch.mean((pred-label)**2)
    #     if e % 10 == 0:
    #         print(loss)
    #     loss.backward()
    #     optm.step()
