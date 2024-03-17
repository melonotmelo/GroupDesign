from models.omniarch import OmniArch
import torch


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    model_ = OmniArch(
        hidden_dim=128,
        patch_size=16,
        num_hidden_layers=12,
        num_t=10,
        img_size=64,
        channels=4,
        phy_pred_method='step',
        num_heads=8,
        dropout_rate=0.1
    ).to(device)
