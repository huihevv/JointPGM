import torch
import torch.nn as nn
import math


class SeriesEmbedding(nn.Module):  
    def __init__(self, L, N, embed_size, latent_size, individual):
        super(SeriesEmbedding, self).__init__()
        self.L = L
        self.N = N
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.individual = individual

        if self.individual is True:
            self.embed = nn.ModuleList()
            self.gate_layer_1 = nn.ModuleList()
            self.mu_h_hat_layer = nn.ModuleList()
            self.var_h_hat_layer = nn.ModuleList()
            for _ in range(self.N):
                self.embed.append(nn.Linear(self.L, self.embed_size))
                self.gate_layer_1.append(nn.Sigmoid())
                self.mu_h_hat_layer.append(nn.Linear(self.embed_size, self.latent_size))
                self.var_h_hat_layer.append(nn.Linear(self.embed_size, self.latent_size))
        else:
            self.embed = nn.Linear(self.L, self.embed_size)
            self.gate_layer_1 = nn.Sigmoid()  
            self.mu_h_hat_layer = nn.Linear(self.embed_size, self.latent_size)
            self.var_h_hat_layer = nn.Linear(self.embed_size, self.latent_size) 


    def forward(self, x, time_x):
        x = x.permute(0, 2, 1)     
        if self.individual is True:
            h = torch.zeros([x.size(0), self.N, self.embed_size], dtype=x.dtype).to(x.device)
            h_hat = torch.zeros([x.size(0), self.N, self.embed_size], dtype=x.dtype).to(x.device)
            mu_h_hat = torch.zeros([x.size(0), self.N, self.latent_size], dtype=x.dtype).to(x.device)
            var_h_hat = torch.zeros([x.size(0), self.N, self.latent_size], dtype=x.dtype).to(x.device)
            for i in range(self.N):
                # forward process
                h[:, i, :] = self.embed[i](x[:, i, :])
                # temporal gate
                h_hat[:, i, :] = self.gate_layer_1[i](time_x[:, i, :]) * h[:, i, :]
                # vae
                mu_h_hat[:, i, :] = self.mu_h_hat_layer[i](h_hat[:, i, :])
                var_h_hat[:, i, :] = self.var_h_hat_layer[i](h_hat[:, i, :])
        else:
            # forward process
            h = self.embed(x)
            # temporal gate
            h_hat = self.gate_layer_1(time_x) * h 
            # vae
            mu_h_hat = self.mu_h_hat_layer(h_hat)
            var_h_hat = self.var_h_hat_layer(h_hat)
        return h, h_hat, mu_h_hat, var_h_hat
    
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, time_features, embed_size):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'q': 1, 'm': 2, 'w': 3, 'd': 6, 'h': 7, 't': 8, 's': 9}
        d_inp = freq_map[time_features]
        # self.embeddings = nn.Parameter(torch.randn(1, embed_size))
        self.embed = nn.Linear(d_inp, embed_size, bias=False)

    def forward(self, x):
        return self.embed(x)
    
class GaussianFourierFeatureTransform(nn.Module):
    def __init__(self, input_dim, fourier_feat_size, sigma_s):
        super().__init__()
        self.input_dim = input_dim
        self.fourier_feat_size = fourier_feat_size
        self.sigma_s = sigma_s

        n_scale_feats = fourier_feat_size // (2 * len(sigma_s))
        assert n_scale_feats * 2 * len(sigma_s) == fourier_feat_size, \
            f"fourier_feat_size: {fourier_feat_size} must be divisible by 2 * len(scales) = {2 * len(sigma_s)}"
        B_size = (input_dim, n_scale_feats)
        B = torch.cat([torch.randn(B_size) * scale for scale in sigma_s], dim=1)
        self.register_buffer('B', B)

    def forward(self, x):
        assert x.dim() >= 2, f"Expected 2 or more dimensional input (got {x.dim()}D input)"
        time, dim = x.shape[-2], x.shape[-1]

        assert dim == self.input_dim, \
            f"Expected input to have {self.input_dim} channels (got {dim} channels)"

        x = torch.einsum('... t n, n d -> ... t d', [x, self.B])
        x = 2 * math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    
class EmbedLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x):
        out = self._layer(x)
        return self.norm(out)

    def _layer(self, x):
        return self.dropout(torch.relu(self.linear(x)))
    
class CoordEmbedding(nn.Module):
    def __init__(self, n_coords_layers, embed_size, fourier_feat_size, sigma_s, dropout=0.1):
        super().__init__()
        self.features = GaussianFourierFeatureTransform(1, fourier_feat_size, sigma_s)
        layers = [EmbedLayer(fourier_feat_size, embed_size, dropout=dropout)] + \
                 [EmbedLayer(embed_size, embed_size, dropout=dropout) for _ in range(n_coords_layers - 1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return self.layers(x)

