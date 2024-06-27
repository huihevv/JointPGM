import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from model.embedding import SeriesEmbedding, TimeFeatureEmbedding, CoordEmbedding
import numpy as np

class jointpgm(nn.Module):
    def __init__(self, args):
        super(jointpgm, self).__init__()
        freq_map = {'q': 1, 'm': 2, 'w': 3, 'd': 6, 'h': 7, 't': 8, 's': 9}
        d_inp = freq_map[args.time_features]
        self.N = args.N_series
        self.B = args.batch_size
        self.L = args.seq_len
        self.H = args.pred_len
        self.device = args.device
        self.embed_size = args.embed_size
        self.latent_size = args.latent_size
        self.temp = args.temp
        self.gc_deep = args.gc_deep
        self.temporal_index = args.temporal_index
        self.individual = args.individual
        self.alpha = args.alpha
        self.SeriesEmbedding = SeriesEmbedding(self.L, self.N, self.embed_size, self.latent_size, self.individual)
        self.TimeFeatureEmbedding = TimeFeatureEmbedding(args.time_features, self.embed_size)
        self.CoordEmbedding = CoordEmbedding(args.n_coords_layers, args.embed_size, args.fourier_feat_size, args.sigma_s)
        self.map_layer = nn.Linear(self.L + self.H, self.L)
        self.gate_layer_0 = nn.Linear(self.L, self.N)
        self.weight_key = nn.Parameter(torch.zeros(size=(self.latent_size, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.latent_size, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5)
        self.mu_t_layer = nn.Linear(self.embed_size, self.latent_size)
        self.var_t_layer = nn.Linear(self.embed_size, self.latent_size)
        self.mlp = nn.Linear((self.gc_deep + 1), 1)
        self.mu_h_tilde_layer = nn.Linear(self.embed_size, self.latent_size)
        self.var_h_tilde_layer = nn.Linear(self.embed_size, self.latent_size)
        self.inference_layer = nn.Linear(self.latent_size, self.latent_size)
        self.mu_z_t_hat_layer = nn.Linear(self.latent_size, self.latent_size)
        self.var_z_t_hat_layer = nn.Linear(self.latent_size, self.latent_size)
        self.decoder_r = nn.Sequential(
            nn.Linear(self.latent_size, self.embed_size),
            nn.LeakyReLU(),
            nn.Linear(self.embed_size, self.L)
        )
        self.decoder_f = nn.Sequential(
            nn.Linear(self.latent_size, self.embed_size),
            nn.LeakyReLU(),
            nn.Linear(self.embed_size, self.H)
        )


    def get_coords(self, L, H):
        coords = torch.linspace(0, 1, L+H)
        # reshape
        return rearrange(coords, 't -> 1 t 1')
    
    def gumbel_softmax(self, x, temp, hard=True, eps=1e-10):
        U = torch.rand(x.size()).to(self.device)
        gumbel_sample = -torch.autograd.Variable(torch.log(-torch.log(U + eps) + eps))
        gumbel_sample = x + gumbel_sample
        gumbel_sample = F.softmax(gumbel_sample / temp, dim=-1)
        y_soft = gumbel_sample
        if hard:
            shape = x.size()
            _, k = y_soft.data.max(-1)
            y_hard = torch.zeros(*shape).to(self.device)
            y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
            y = torch.autograd.Variable(y_hard - y_soft.data) + y_soft
        else:
            y = y_soft
        return y
    
    def self_graph_attention(self, x):
        key = torch.matmul(x, self.weight_key)
        query = torch.matmul(x, self.weight_query)
        data = key.repeat(1, 1, self.N).view(x.size(0), self.N * self.N, 1) + query.repeat(1, self.N, 1)
        data = data.squeeze(2)
        data = data.view(x.size(0), self.N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention


    def discrete_graph(self, x, is_eval):
        # StemGNN
        learned_graph = self.self_graph_attention(x)
        learned_graph = (learned_graph + 1) / 2.
        learned_graph = torch.stack([learned_graph, 1-learned_graph], dim=-1)
        learned_graph = learned_graph.reshape(self.N * self.N, -1)

        # gumbel softmax
        if is_eval:
            adj = self.gumbel_softmax(learned_graph, self.temp, hard=True)
        else:
            adj = self.gumbel_softmax(learned_graph, self.temp, hard=True)
        
        adj = adj[:, 0].clone().reshape(self.N, -1)
        mask = torch.eye(self.N, self.N).bool().to(self.device)
        adj.masked_fill_(mask, 0)
        return adj
    
    def graph_convolution(self, x, adj, gc_deep):
        H = x
        H_0 = x.unsqueeze(dim=-1)
        for _ in range(gc_deep):
            # H = torch.einsum("bnd,bnn->bnd", (H, adj))
            H = torch.einsum("bnd,nn->bnd", (H, adj))
            H_0 = torch.cat((H_0, H.unsqueeze(dim=-1)), dim=3)
        H_0 = self.mlp(H_0).squeeze(dim=3)
        return H_0
    
    def reparameterization(self, mu, var):
        return (torch.randn_like(torch.exp(0.5 * var)) * torch.exp(0.5 * var) + mu)


    def forward(self, batch_x, time_x, is_eval):
        # time factor encoder (TFE)
        if self.temporal_index is True:
            # order features
            # time_x [1, L+H, 1] -> [B, L, D]
            coords = self.get_coords(self.L, self.H).to(self.device)
            time_x = repeat(self.CoordEmbedding(coords), '1 t d -> b t d', b=time_x.size(0))
            time_x = self.map_layer(time_x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            # timestamp features
            # time_x [B, L, Q] -> [B, L, D]
            time_x = self.TimeFeatureEmbedding(time_x)
        
        time_x = self.gate_layer_0(time_x.permute(0, 2, 1)).permute(0, 2, 1)
        mu_t = self.mu_t_layer(time_x)
        var_t = self.var_t_layer(time_x)
        z_t = self.reparameterization(mu_t, var_t)

        # independence-based series encoder (ISE)
        h, h_hat, mu_h_hat, var_h_hat = self.SeriesEmbedding(batch_x, time_x)
        z_hat = self.reparameterization(mu_h_hat, var_h_hat)
 

        adj = self.discrete_graph(z_hat, is_eval)
        h_tilde = self.graph_convolution(h_hat, adj, self.gc_deep)
        mu_h_tilde = self.mu_h_tilde_layer(h_tilde)
        var_h_tilde = self.var_h_tilde_layer(h_tilde)
        z_tilde = self.reparameterization(mu_h_tilde, var_h_tilde)
        z = self.alpha * z_hat + (1 - self.alpha) * z_tilde
        mu_z = self.alpha * mu_h_hat + (1 - self.alpha) * mu_h_tilde
        var_z = self.alpha * var_h_hat + (1 - self.alpha) * var_h_tilde

        # dynamic inference (DI)
        z_t_hat = self.inference_layer(z)
        mu_z_t_hat = self.mu_z_t_hat_layer(z_t_hat)
        var_z_t_hat  = self.var_z_t_hat_layer(z_t_hat)

        # decoder
        y = self.decoder_f[2](self.decoder_f[1](self.decoder_f[0](z))+h).permute(0, 2, 1)
        x = self.decoder_r[2](self.decoder_r[1](self.decoder_r[0](z))+h).permute(0, 2, 1)

   
        return x, y, mu_t, var_t, mu_z_t_hat, var_z_t_hat, mu_z, var_z
