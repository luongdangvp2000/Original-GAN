import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader



class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_features, out_features, normalize=True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            
        )