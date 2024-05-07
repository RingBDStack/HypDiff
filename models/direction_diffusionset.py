import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from manifolds.poincare import PoincareBall
from torch_geometric.nn import GCNConv
from tqdm.auto import tqdm

from layers.layers import FermiDiracDecoder

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


class gcndec(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(gcndec, self).__init__()
        self.conv1 = GCNConv(input_dim + 1, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.conv4 = GCNConv(256, 128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x, adj, t):
        x = x.to(torch.float32)
        x = torch.cat((x, t.unsqueeze(1)), dim=1).float()
        x = self.conv1(x, adj)
        x = F.gelu(x)
        x = self.conv2(x, adj)
        x = F.gelu(x)
        x = self.conv3(x, adj)
        x = F.gelu(x)
        x = self.conv4(x, adj)
        x = F.gelu(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        return x


class mlp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim + 1, 800)
        self.fc2 = nn.Linear(800, 900)
        self.fc3 = nn.Linear(900, 800)
        self.fc4 = nn.Linear(800, 700)
        self.fc5 = nn.Linear(700, output_dim)

    def forward(self, x, t):
        x = x.to(torch.float32)
        t_ex = t.unsqueeze(1).unsqueeze(2)
        t_ex_expanded = t_ex.expand(-1, x.shape[1], 1)
        x = torch.cat((x, t_ex_expanded), dim=-1).float()
        # x = torch.cat((x, t.unsqueeze(1)), dim=2).float()
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        x = self.fc4(x)
        x = F.gelu(x)
        x = self.fc5(x)

        return x


def get_alphas(timesteps):
    def linear_beta_schedule(timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

    betas = linear_beta_schedule(timesteps)
    alphas = 1. - betas

    return alphas, betas


class diffusionset(nn.Module):
    def __init__(self, args, in_dim, out_dim):
        super(diffusionset, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.prior = 1
        self.gcn = gcndec(in_dim, out_dim)
        self.gcn.to(device)
        self.mlp = mlp(in_dim, out_dim)
        self.mlp = self.mlp.to(device)
        self.num_timesteps = 200
        self.manifold = PoincareBall()
        self.alphas, betas = get_alphas(self.num_timesteps)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                1. - self.alphas_cumprod)
        self.alphas = self.alphas.cuda()
        self.alphas_cumprod = self.alphas_cumprod.cuda()
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.cuda()
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.cuda()
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.cuda()
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.cuda()
        self.posterior_variance = self.posterior_variance.cuda()
        self.minus_alphas_cumprod = self.minus_alphas_cumprod.cuda()
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.cuda()
        self.posterior_mean_coef1 = self.posterior_mean_coef1.cuda()
        self.posterior_mean_coef2 = self.posterior_mean_coef2.cuda()
        self.c = 1.0
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)

    def exists(self, x):
        return x is not None

    def default(self, val, d):
        if self.exists(val):
            return val
        return d() if callable(d) else d

    def cal_u0(self, data):

        u = []
        for x in data:
            x = self.manifold.logmap0(x, c=1.0)
            u0 = torch.mean(x, dim=0)
            u0 = self.manifold.expmap0(u0, c=1.0)
            u.append(u0)
        # u0=self.manifold.expmap0(u0,c=1.0)
        return u

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def normalization(self, x):
        mean_x = torch.mean(x)
        std_x = torch.std(x)
        n1 = (x - mean_x) / std_x
        return n1

    def renormal(self, x, h):
        mean = torch.mean(h)
        std = torch.std(h)
        return x * std + mean

    def tran_direction(self, direction_vector, gaussian_point, labels):
        # Given a vector
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # tran=[]
        for i, noise in enumerate(gaussian_point):
            for j, p_noise in enumerate(noise):
                transformed_vector = torch.sign(direction_vector[i][labels[i][j]])
                # transformed_vector= -1 * transformed_vector
                gaussian_point[i][j] = p_noise * transformed_vector

        return gaussian_point

    # def tran_direction(self, direction_vector, gaussian_point, labels):
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     transformed_vectors = [-1 * torch.sign(direction_vector[i]) for i in labels]
    #     gaussian_point *= torch.stack(transformed_vectors)
    #     # gaussian_point *= transformed_vectors[:, None]
    #     # Broadcasting to multiply with gaussian_point
    #     return gaussian_point

    def get_alphas2(self, timesteps):
        def linear_beta_schedule(timesteps):
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

        betas = linear_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_minus = torch.sqrt(1. - alphas_cumprod)
        return torch.sqrt(alphas_cumprod), alphas_minus

    def x_tran(self, x0, labels, u0):
        for i, x in enumerate(x0):
            u = u0[labels[i]]
            x = self.manifold.logmap(u, x, c=1.0)
        return x0

    def getpri(self, t):
        c_num = np.sqrt(self.c)
        T = 500
        out = self.prior * torch.tanh(c_num * t / T)
        b, *_ = t.shape
        return out.reshape(b, *((1,) * (2 - 1)))

    def q_sample(self, x_start, t, direction, noise, labels, u0, restrict=False):
        noise = self.default(noise, lambda: torch.randn_like(x_start))
        if restrict:
            noise = self.tran_direction(direction, noise, labels)
        alphas_cumprod, minus_alphas_cumprod = self.get_alphas2(self.num_timesteps)
        alphas_cumprod = alphas_cumprod.cuda()
        minus_alphas_cumprod = minus_alphas_cumprod.cuda()
        pri_cumprod = self.getpri(t)
        if not restrict:
            return (
                    self.extract(alphas_cumprod, t, x_start.shape) * x_start +
                    self.extract(minus_alphas_cumprod, t, x_start.shape) * noise + pri_cumprod.unsqueeze(1) * x_start
            )
        else:
            return (self.extract(alphas_cumprod, t, x_start.shape) * x_start +
                    self.extract(minus_alphas_cumprod, t, x_start.shape) * noise + pri_cumprod.unsqueeze(1) * x_start)

    def edge(self, adj):
        sparse_tensor = adj
        # values = sparse_tensor._values()
        row_indices = sparse_tensor._indices()[0]  # get row index
        col_indices = sparse_tensor._indices()[1]  # get col index

        # Create edge_index, which is a tensor of size [2, num_edges]
        edge_index = torch.stack([row_indices, col_indices], dim=0)

        # If you need to convert the edge_index to a LongTensor, you can use the following code
        edge_index = edge_index.to(torch.long)
        return edge_index

    def predict_noise_from_start(self, x_t, t, x0):
        prio = self.getpri(t)
        # predicte Gaussian noise from the starting state of the prediction graph
        return (
                (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t + prio.unsqueeze(1) * x_t - x0) / \
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        # Compute the mean and variance of the posterior distribution q(x_t | x_{t-1}).
        posterior_mean = (
                self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, adj, t, x_self_cond=None, graphset=False):
        # predicte the noise or onset state of a sample based on a generative model
        # x=expmap0(x,c=1.0)
        if (graphset):
            model_output = self.mlp(x, t)
        else:
            model_output = self.gcn(x, adj, t)
        # maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        # x=logmap0(x,c=1.0)
        x_start = model_output
        # x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, adj, t, x_self_cond=None, clip_denoised=True, graphset=False):
        # calculate the mean and variance of p(x_t | x_0)
        preds = self.model_predictions(x, adj, t, x_self_cond, graphset)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        # model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)

        return preds.pred_noise, preds.pred_x_start

    def p_sample(self, x, h, adj, direction, labels, u0, t: int, x_self_cond=None, graphset=False):
        # sample according to p(x_t | x_0)
        b, *_, device = *x.shape, self.device
        # x=self.getpri(t)*h +self.extract(self.minus_alphas_cumprod, batched_times, x_start.shape) * x
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        # x=self.getpri(batched_times)*h +self.extract(self.minus_alphas_cumprod, batched_times, x.shape) * x
        noise, x_start = self.p_mean_variance(x=x, adj=adj, t=batched_times, x_self_cond=x_self_cond,
                                              clip_denoised=True, graphset=graphset)
        # noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        # pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        if (t > 0):
            t_tmp = t - 1
        else:
            t_tmp = 0
        t2 = torch.full((b,), t_tmp, device=device, dtype=torch.long)
        # print(h.shape)
        noise_tmp = noise
        # pred_img_t=self.extract(self.alphas_cumprod, batched_times, x_start.shape) * x_start +self.extract(self.minus_alphas_cumprod, batched_times, x_start.shape) * noise
        pred_img_t = self.q_sample(x_start=x_start, t=t2, direction=direction, noise=noise_tmp, labels=labels, u0=u0,
                                   restrict=True)

        return pred_img_t, x_start

    def p_sample_loop(self, x, adj, lables, u0, return_all_timesteps=False, graphset=True):
        shape = x.shape
        # print(x)
        batch, device = shape[0], self.device
        img = torch.randn(shape, device=device)
        direction = []
        for u in u0:
            direction.append(self.manifold.logmap0(u, c=1.0))
        # img=self.tran_direction(direction,img,lables)
        self_condition = None
        x_start = None
        h = x
       # h = self.manifold.logmap0(h, c=1.0)
        x_start=h
        # print(x_start.shape)
        for i, x_emb in enumerate(x_start):
            for j, xemb in enumerate(x_start[i]):
                # print(x_emb.shape)
                x_start[i][j] = self.manifold.logmap(u0[i][lables[i][j]], xemb, c=1.0)
        # for i, h_data in enumerate(h):
        #     h[i] = self.manifold.logmap(u0[lables[i]], h_data, c=1.0)
        h2 = self.normalization(x_start)

        img = img + h2
        imgs = [img]
        if (graphset):
            adj = 1
        else:
            adj = self.edge(adj)
        # for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self_condition else None
            img = img
            img, x_start = self.p_sample(img, h2, adj, direction, lables, u0, t, self_cond, graphset)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        # ret = self.unnormalize(ret)
        # ret = self.manifold.expmap0(ret, c=1.0)
        u20 = u0
        ret = self.renormal(ret, h)
        # ret=h
        # ret = self.manifold.expmap0(ret, c=1.0)
        # for i,x_pred in enumerate(ret):
        #     #ret[i]=self.manifold.expmap(u0[lables[i]],x0_pred,c=1.0)
        #     ret[i]=self.manifold.expmap(ret[i],u20[lables[i]],c=1.0)
        for i, x_emb in enumerate(ret):
            for j, xemb in enumerate(ret[i]):
                # print(x_emb.shape)
                ret[i][j] = self.manifold.expmap(xemb, u0[i][lables[i][j]], c=1.0)
        return ret

    def forward(self, x, adj, labels, u0, restrict=False, graphset=False):
        x = x.to(float)
        h = x
        b = h.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=h.device).long()
        noise = torch.randn_like(h)
        noise.to(h.device)
        direction = []
        for u in u0:
            direction.append(self.manifold.logmap0(u, c=1.0))
        if not restrict:
            x_start = self.manifold.logmap0(x, c=1.0)
        else:
            x_start = x
            # print(x_start.shape)
            for i, x_emb in enumerate(x_start):
                for j, xemb in enumerate(x_start[i]):
                    # print(x_emb.shape)
                    x_start[i][j] = self.manifold.logmap(u0[i][labels[i][j]], xemb, c=1.0)
        x_start = self.normalization(x_start)
        xt = self.q_sample(x_start=x_start, t=t, direction=direction, noise=noise, labels=labels, u0=u0, restrict=True)

        t = t / self.num_timesteps
        if (graphset):
            out = self.mlp(xt, t)
        else:
            edge_index = self.edge(adj)
            out = self.gcn(xt, edge_index, t)
        x_start = x_start.to(torch.float32)
        loss = F.mse_loss(out, x_start)
        return loss
