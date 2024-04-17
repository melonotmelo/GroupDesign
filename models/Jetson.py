import math

import einops
import torch
from torch import nn
import json
from models.revin import RevIN
from models.omniarch import *
from models.transformer import *
import socket
import time

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
        self.encoder = OmniArchEncoder(**config)
        self.backbone = Transformer(**config)
        self.decoder = OmniArchDecoder(**config)
        # init params

    def forward(self, phy_fls, attention_mask=None):
        # phy_fls: B T C X (Y) (Z)
        phy_fls = einops.rearrange(phy_fls, 'b t c ... -> b t ... c')
        phy_fls, mean, stdev = self.revin(phy_fls, 0, 0, "norm")
        embedding = self.encoder(phy_fls)
        return embedding, mean, stdev


def run_edge():
    model_ = Jetson(
        hidden_dim=128,
        patch_size=16,
        num_hidden_layers=12,
        num_t=10,
        train_img_size=32,
        eval_img_size=32,
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
    with torch.no_grad():  # 不计算梯度，节省计算资源
        pred = model_(phy_fls_2d)
    pred_list = pred.tolist()
    f2 = open('data_to_send.json', 'w')
    json.dump(pred_list, f2)
    f2.close()
    server_host = '192.168.31.60'
    server_port = 12345
    buffer_size = 6400
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((server_host, server_port))
        start_time = time.time()
        with open('data_to_send.json', 'rb') as file:
            while True:
                bytes_read = file.read(buffer_size)
                s.sendall(bytes_read)
                if not bytes_read:
                    s.sendall(b'EOF')
                    break  # 文件结束
            file.close()
        response = s.recv(1024)
        end_time = time.time()
        print(f"Received response: {response.decode()} in {end_time - start_time} seconds")
