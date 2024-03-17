import os

import torch
from determined.pytorch import DataLoader, PyTorchTrial, LRScheduler, PyTorchTrialContext
from models.omniarch import OmniArch, pass_batch
from model_hub.huggingface import _config_parser as hf_parse
from model_hub.huggingface._trial import build_default_lr_scheduler
from torch.cuda.amp import GradScaler
from others.data_information import datafile_paths

from dataset.omniarch_dataset import MixedOmniArchDataset, OmniArchSampler
from utils.s3 import download
from determined import pytorch as det_pytorch
from typing import Dict, Any


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UniPDETrial(PyTorchTrial):

    @staticmethod
    def clip_grads(params):
        torch.nn.utils.clip_grad_norm_(params, 0.01)

    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.hparams = self.context.get_hparams()
        config = self.hparams
        is_continue = False if self.hparams["continue"] is None else self.hparams["continue"]
        model = OmniArch(**config).to(device)
        if is_continue:
            model_ckpt_folder = self.hparams["checkpoint_folder"]
            if not os.path.exists(f'state_dict.pth'):
                download('det-ckpt-storage', f'state_dict.pth',
                         f's3_store/{model_ckpt_folder}/state_dict.pth')
            model.load_state_dict(torch.load(f'state_dict.pth')['models_state_dict'][0])
            print("load over")
        scaler = GradScaler()
        self.context.wrap_scaler(scaler)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.hparams["lr"])
        self.model = self.context.wrap_model(model)
        self.optimizer = self.context.wrap_optimizer(
            optimizer, fp16_compression=True
        )
        num_examples = self.hparams['scheduler']['records_per_epoch']
        lr_circle_epoch = self.hparams['scheduler']['epochs']
        batch_size = self.hparams['scheduler']['batch_size']
        batches = int(num_examples / batch_size * lr_circle_epoch)
        scheduler_kwargs = hf_parse.LRSchedulerKwargs(num_training_steps=batches,
                                                      lr_scheduler_type="cosine", num_warmup_steps=50)
        self.lr_scheduler = self.context.wrap_lr_scheduler(
            build_default_lr_scheduler(self.optimizer, scheduler_kwargs),
            step_mode=LRScheduler.StepMode.STEP_EVERY_BATCH
        )

        self.attn_mask = None

    def train_batch(self, batch, **kwargs):
        phy_fls, tp = batch
        print(f"train: {phy_fls.shape}, resolution: {phy_fls.shape[-1]}, type: {tp.tolist()}")
        with torch.autocast(device_type="cuda"):
            loss_info = pass_batch(self.model, phy_fls, tp)
        self.context.backward(loss_info['loss'].float())
        self.context.step_optimizer(self.optimizer, clip_grads=self.clip_grads)
        self.lr_scheduler.step()
        print(loss_info)
        return {"lr": self.lr_scheduler.get_last_lr(), **loss_info}

    def evaluate_batch(self, batch: det_pytorch.TorchData, batch_idx: int) -> Dict[str, Any]:
        phy_fls, tp = batch
        print(f"eval {batch_idx}: {phy_fls.shape}, resolution: {phy_fls.shape[-1]}, type: {tp.tolist()}")
        with torch.autocast(device_type="cuda"):
            loss_info = pass_batch(self.model, phy_fls, tp, self.attn_mask)
        print(loss_info)
        return loss_info

    def build_training_data_loader(self) -> DataLoader:
        dataset = MixedOmniArchDataset("train", num_t=10, datafile_paths=datafile_paths,
                                       x_size=self.hparams['train_img_size'])
        print(len(dataset))
        return DataLoader(dataset, batch_size=self.hparams['train_batch_size'], num_workers=0, shuffle=True)

    def build_validation_data_loader(self) -> DataLoader:
        dataset = MixedOmniArchDataset("val", num_t=10, datafile_paths=datafile_paths,
                                       x_size=self.hparams['eval_img_size'])
        print(len(dataset))
        return DataLoader(dataset, batch_size=self.hparams['eval_batch_size'], num_workers=0)
