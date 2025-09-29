from typing import Optional, Union

import numpy as np
import torch

beta_1 = 1e-4
beta_T = 0.02
num_train_timesteps = 1000

s = 0.008
timesteps = torch.linspace(0, num_train_timesteps, num_train_timesteps+1)
print("timesteps:", timesteps)
angles = ((timesteps / num_train_timesteps) + s) / (1 + s) * (torch.pi / 2)
print("angles:", angles)
alphas_cumprod = torch.cos(angles) ** 2
print("alphas_cumprod1:", alphas_cumprod)
alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
print("alphas_cumprod2:", alphas_cumprod)
betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
print("betas:", betas)