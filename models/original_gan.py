import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader



class Generator(nn.Module):
    def __init__(self, img_size, latent_size, hidden_size):
        super().__init__()
        self.latent_size = latent_size
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU,
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, img_size),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, img_size, hidden_size):
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.model = nn.Sequential(
            nn.Linear(img_size, hidden_size),
            nn.LeakyReLu(),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLu(),
            nn.Dropout(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, self.img_size)
        return self.model(x).clamp(1e-9)