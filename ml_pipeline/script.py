## Need to module load cuda/12.2.0 !!!

import sys 
print (sys.executable)

import os
import torch
# import wandb
# from torchinfo import summary
# from torch.utils.data import random_split, DataLoader
import torch.nn as nn
# import argparse

from models import GenomicClassifier, VariationalAutoEncoderConv, VariationalAutoEncoderResNet

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = VariationalAutoEncoderResNet(input_dim=(3, 224, 224), latent_dim=128)
# model = GenomicClassifier(input_dim=(1, 980, 980), latent_dim=200000, num_classes=117670).to(device)

out = model(torch.randn(1, 3, 224, 224))

print(model)

print("Done")