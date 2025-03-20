import os
import random
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class CustomMSELoss:
    def __call__(self, recon_x, x):
        MSE = F.mse_loss(
            recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        ).sum(dim=-1)
        return MSE.mean(dim=0)

class VAELoss:
    def __call__(self, recon_x, x, mu, logvar):
        # MSE = F.mse_loss(
        #     recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1), reduction="none"
        # ).sum(dim=-1)
        MSE = nn.MSELoss(reduction='mean')(recon_x, x)
        # BCE = F.binary_cross_entropy(recon_x.reshape(x.shape[0], -1), x.reshape(x.shape[0], -1))
        # KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim=0)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        beta = 0.000000001
        print("MSE: ", MSE)
        print("KLD: ", KLD)
        return MSE + beta*KLD

def set_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    eval("setattr(torch.backends.cudnn, 'deterministic', True)")
    eval("setattr(torch.backends.cudnn, 'benchmark', False)")
    os.environ["PYTHONHASHSEED"] = str(seed)

def save_checkpoint(model, optimizer, epoch, filename="checkpoint.pth.tar"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']