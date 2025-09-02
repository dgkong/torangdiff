# referred https://huggingface.co/blog/annotated-diffusion

import torch
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class DDPM:
    def __init__(self, T, device):
        self.T = T
        self.betas = cosine_beta_schedule(T).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.alphas_cumprod[t].view(-1,1,1)
        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1-a_bar) * noise
        return x_t, noise

    def p_sample(self, x_t, t, eps_hat):
        betas_t = self.betas[t].view(-1,1,1)
        alphas_t = self.alphas[t].view(-1,1,1)
        a_bar = self.alphas_cumprod[t].view(-1,1,1)

        mean = (1/torch.sqrt(alphas_t)) * (x_t - (betas_t/torch.sqrt(1-a_bar)) * eps_hat)
        
        if (t==0).all():
            return mean
        z = torch.randn_like(x_t)
        sigma = torch.sqrt(self.posterior_variance[t].view(-1,1,1))
        return mean + sigma * z
