import argparse
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from transformers import VivitForVideoClassification, VivitConfig
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
import copy
import math
# from utils import CosineAnnealingWithWarmupLR, concat_all_gather
from pytorch_lightning.loggers import TensorBoardLogger

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
# from pytorch_lightning.strategies import DeepSpeedStrategy

# dist.init_process_group("nccl")

# local_rank = int(os.environ["LOCAL_RANK"])
# global_rank = int(os.environ["RANK"])
batch_size = 2

class CosineAnnealingWithWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, T_max, eta_min=0, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmupLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = self.last_epoch / self.warmup_steps
        else:
            scale = (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps)/ (self.T_max - self.warmup_steps))) / 2
        
        return [self.eta_min + (base_lr - self.eta_min) * scale for base_lr in self.base_lrs]
    
    
class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


# Define MoCo class
class MoCo(pl.LightningModule):
    def __init__(self, encoder='VivitForVideoClassification', out_dim=256, mlp_dim=4096, tau=0.2, mu=0.99, lr=1.5e-4, weight_decay=0.1, warmup_steps=1, max_steps=10, num_classes=10, pre_trained='pretrained'):
        super(MoCo, self).__init__()
        
        self.save_hyperparameters()
        self.training_losses = []  # Initialize list to store training loss

        # Load ViViT model
        configuration = VivitConfig(image_size=112, num_labels=num_classes, ignore_mismatched_sizes=True)
        self.encoder = VivitForVideoClassification(config=configuration)
        print(self.encoder.config)


        # Get hidden dimension from classifier layer
        hidden_dim = self.encoder.config.hidden_size
        
        # Replace classifier with a MLP for contrastive learning
        self.encoder.classifier = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False) 
        )
        
        # Build momentum encoder
        self.momentum_encoder = copy.deepcopy(self.encoder)
        
        # Build predictor
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, mlp_dim, bias=False),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)
        )
        
        # Stop gradient in momentum encoder
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
            
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineAnnealingWithWarmupLR(optimizer, self.hparams.warmup_steps, self.hparams.max_steps)
                
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def forward(self, x):
        return self.encoder(x).logits

    def contrastive_loss(self, q, k):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        k = concat_all_gather(k)
        
        N = q.shape[0]  # batch size per GPU        
        logits = q @ k.T
        labels = torch.arange(N, dtype=torch.long).to(device=self.device)
        
        loss = nn.functional.cross_entropy(logits / self.hparams.tau, labels)
        
        return 2 * self.hparams.tau * loss
        
    @torch.no_grad()
    def _update_momentum_encoder(self, batch_idx):
        # Update mu with a cosine schedule
        current_step = self.current_epoch * self.trainer.num_training_batches + batch_idx
        mu = (1 - (1 + math.cos(math.pi * current_step / self.hparams.max_steps)) / 2) * (1-self.hparams.mu) + self.hparams.mu
        
        # Update momentum encoder's parameters
        for param, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * mu + param.data * (1. - mu)
        
    def training_step(self, batch, batch_idx):
        (x1, x2), _ = batch
        
        self._update_momentum_encoder(batch_idx)
        
        # encoder forward pass
        encoder_output1 = self.encoder(x1).logits
        encoder_output2 = self.encoder(x2).logits
        
        # Extract logits or features from the encoder outputs
        q1 = self.predictor(encoder_output1)
        q2 = self.predictor(encoder_output2)
        
        # momentum encoder forward pass
        momentum_output1 = self.momentum_encoder(x1).logits
        momentum_output2 = self.momentum_encoder(x2).logits
        
        # Extract logits or features from the momentum encoder outputs
        k1 = momentum_output1
        k2 = momentum_output2
        
        # calculate MoCo contrastive loss
        loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        self.training_losses.append(loss.item())  # Store the loss
        self.log('MoCo-v3 loss', loss)
        
        return loss

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_path, size, transforms_list):
        self.size = size
        df = pd.read_csv(csv_path)
        self.image_paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transforms = transforms_list

    def __getitem__(self, index):
        path = self.image_paths[index]
        label = torch.tensor(self.labels[index], dtype=torch.long)
        
        # Open image
        try:
            img = Image.open(path).convert('L')  # Convert to grayscale
        except Exception as e:
            raise RuntimeError(f"Error loading image at {path}: {e}")
        
        # Initialize list to store cropped image sections
        video = []
        count = 0
        
        # Create cropped sections
        for i in range(5):  # 5 rows
            for j in range(6):  # 6 columns
                left = 29 * j
                top = 29 * i
                right = left + 28
                bottom = top + 28
                
                # Crop, resize, and append the section
                cropped_img = img.crop((left, top, right, bottom)).resize(self.size)
                video.append(np.array(cropped_img))
                count += 1
                
                if count == 28:
                    break
            if count == 28:
                for _ in range(4):
                    cropped_img = img.crop((left, top, right, bottom)).resize(self.size)
                    video.append(np.array(cropped_img))
                break
        
        # Normalize the video array
        video = np.array(video)
        video = (video - video.min()) / (video.max() - video.min())
        video = np.stack([video] * 3, axis=1)  # Add channel dimension
        
        # Convert to tensor
        video_tensor = torch.from_numpy(video).float()
        
        # Apply two different transformations
        x1 = video_tensor.clone()
        x2 = video_tensor.clone()

        for t in self.transforms:
            x1 = t(x1)
            x2 = t(x2)
        
        return (x1, x2), label

    def __len__(self):
        return len(self.labels)


train_path = '/shared/home/v_nishanth_artham/local_scratch/CryoET/3d_resnet/data_split/train_real_split.csv'
val_path = '/shared/home/v_nishanth_artham/local_scratch/CryoET/3d_resnet/data_split/val_real_split.csv'
test_path = '/shared/home/v_nishanth_artham/local_scratch/CryoET/3d_resnet/data_split/test_real_split.csv'

# Define transformations
transforms_list = [
    transforms.RandomResizedCrop(size=(112, 112), scale=(0.5, 1)),
    transforms.RandomAffine(degrees=(-45, 45), translate=(0.1,0.1), scale=(0.9, 1.1))
]

# Instantiate datasets and dataloaders
train_dataset = CustomDataset(train_path, size=(112, 112), transforms_list=transforms_list)
val_dataset = CustomDataset(val_path, size=(112, 112), transforms_list=transforms_list)
test_dataset = CustomDataset(test_path, size=(112, 112), transforms_list=transforms_list)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=6, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=1, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=1, drop_last=False)
logger = TensorBoardLogger(save_dir='tb_logs', name='pre-train', default_hp_metric=False)

# Instantiate model and trainer
model = MoCo(
    encoder='VivitForVideoClassification',
    out_dim=256,
    mlp_dim=4096,
    tau=0.6,
    mu=0.95,
    lr=7e-4,
    weight_decay=0.99,
    warmup_steps=5,
    max_steps=10,
    pre_trained='none'
)

device_ids = [i for i in range(torch.cuda.device_count())]

# model = nn.DataParallel(model, device_ids=device_ids)
# model = DDP(model, device_ids=[local_rank])
# strategy = DeepSpeedStrategy()
trainer = pl.Trainer(
    devices=[0],
    accelerator='cuda',
    precision="32",  # Mixed precision
    max_epochs=100,
    check_val_every_n_epoch=1,
    logger=logger,
    # strategy=strategy,
    gradient_clip_val=0.5,
    log_every_n_steps=10,
    accumulate_grad_batches=10,
    sync_batchnorm=True,
)

trainer.fit(model, train_loader)


print(model.training_losses)

