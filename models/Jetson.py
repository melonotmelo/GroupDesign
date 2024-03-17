import math

import einops
import torch
from torch import nn

from models.revin import RevIN

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


class Jetson(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.num_t = config['num_t']
        self.patch_size = config['patch_size']
        self.phy_pred_method = config['phy_pred_method']
        # revin
        channels = config['channels']
        self.revin = RevIN(channels)

    def forward(self, phy_fls, attention_mask=None):
        # phy_fls: B T C X (Y) (Z)
        phy_fls = einops.rearrange(phy_fls, 'b t c ... -> b t ... c')
        phy_fls = self.revin(phy_fls, "norm")
        return phy_fls


if __name__ == "__main__":
    model_ = Jetson(
        hidden_dim=128,
        patch_size=16,
        num_hidden_layers=12,
        num_t=10,
        # train_img_size=64,
        # eval_img_size=64,
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
