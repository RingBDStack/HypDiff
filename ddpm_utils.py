import torch
from tqdm import tqdm
import numpy as np

class DDPMSampler():
    def __init__(self, beta_1, beta_T, T, diffusion_fn, device, shape):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_fn  : trained diffusion network
        shape         : data shape
        '''

        self.betas = torch.linspace(start=beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0).to(device=device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.shape = shape
        self.deta = 0.01
        self.diffusion_fn = diffusion_fn
        self.device = device

    def getpri(self, t):
        c_num = np.sqrt(self.c)
        T = 2000
        out = self.deta * torch.tanh(c_num * t / T)
        #print(c_num * t / T)
        b, *_ = t.shape
        return out.reshape(b, *((1,) * (2 - 1)))
    
    def _one_diffusion_step(self, x, direction, restrict, target):
        '''
        x   : perturbated data
        '''
        if not restrict:
            for idx in reversed(range(len(self.alpha_bars))):
                noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
                sqrt_tilde_beta = torch.sqrt(
                    (1 - self.alpha_prev_bars[idx]) / (1 - self.alpha_bars[idx]) * self.betas[idx])
                if target == 'pred_noise':
                    predict_epsilon = self.diffusion_fn(x, idx, get_target=False)
                    mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (
                                x - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon)
                    x = mu_theta_xt + sqrt_tilde_beta * noise
                elif target == 'pred_x0':
                    predict_x0 = self.diffusion_fn(x, idx)
                    predict_epsilon = torch.sqrt(1 / (1 - self.alphas[idx])) * (
                                x - torch.sqrt(self.alphas[idx]) * predict_x0)
                    mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (
                                x - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon)
                    x = mu_theta_xt + sqrt_tilde_beta * noise
                yield x
        else:
            for idx in reversed(range(len(self.alpha_bars))):
                pri = self.getpri(idx)
                noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
                noise = direction * torch.abs(noise)
                sqrt_tilde_beta = torch.sqrt(
                    (1 - self.alpha_prev_bars[idx]) / (1 - self.alpha_bars[idx]) * self.betas[idx])
                if target == 'pred_noise':
                    predict_epsilon = self.diffusion_fn(x, idx)
                    predict_x0 = (x - torch.sqrt(1 - self.alpha_bars[idx]) * direction * torch.abs(predict_epsilon)) / (
                                torch.sqrt(self.alpha_bars[idx]) + self.deta)
                    if (idx > 0):
                        x = x - (torch.sqrt(self.alpha_bars[idx]) * predict_x0 + pri * predict_x0 + torch.sqrt(
                            1 - self.alpha_bars[idx]) * noise) + (torch.sqrt(
                            self.alpha_bars[idx - 1]) * predict_x0 + pri * predict_x0 + torch.sqrt(
                            1 - self.alpha_bars[idx - 1]) * noise)
                elif target == 'pred_x0':
                    predict_x0 = self.diffusion_fn(x, idx)
                    if (idx > 0):
                        x = x - (torch.sqrt(self.alpha_bars[idx]) * predict_x0 + pri * predict_x0 + torch.sqrt(
                            1 - self.alpha_bars[idx]) * noise) + (torch.sqrt(
                            self.alpha_bars[idx - 1]) * predict_x0 + pri * predict_x0 + torch.sqrt(
                            1 - self.alpha_bars[idx - 1]) * noise)

                yield x

    @torch.no_grad()
    def sampling(self, sampling_number, x, only_final=False, restrict=False, target='pred_noise'):
        '''
        sampling_number : a number of generation
        only_final      : If True, return is an only output of final schedule step
        '''
        direction = torch.sign(x)
        sample = torch.randn([sampling_number, *self.shape]).to(device=self.device).squeeze()
        sampling_list = []

        final = None
        for idx, sample in enumerate(tqdm(self._one_diffusion_step(sample, direction, restrict, target))):
            final = sample
            if not only_final:
                sampling_list.append(final)

        return final if only_final else torch.stack(sampling_list)
