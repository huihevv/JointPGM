import torch
import torch.nn as nn
from einops import rearrange, repeat
from model.embedding import TimeFeatureEmbedding, CoordEmbedding


class DSMMTS(nn.Module):
    def __init__(self, args, mixing_model):
        super(DSMMTS, self).__init__()
        freq_map = {'q': 1, 'm': 2, 'w': 3, 'd': 6, 'h': 7, 't': 8, 's': 9}
        d_inp = freq_map[args.time_features]
        self.N = args.N_series
        self.B = args.batch_size
        self.L = args.seq_len
        self.H = args.pred_len
        self.label_len = args.label_len
        self.device = args.device
        self.d_model = args.d_model
        self.temporal_index = args.temporal_index
        self.TimeFeatureEmbedding = TimeFeatureEmbedding(args.time_features, self.d_model)
        self.CoordEmbedding = CoordEmbedding(args.n_coords_layers, args.d_model, args.fourier_feat_size, args.sigma_s)
        self.map_layer = nn.Linear(self.L + self.H, self.L)
        self.mu_t_layer = nn.Linear(self.d_model, self.d_model)
        self.var_t_layer = nn.Linear(self.d_model, self.d_model)
        self.gate_layer_1 = nn.Sigmoid()  
        self.mu_z_layer = nn.Linear(self.d_model, self.d_model)
        self.var_z_layer = nn.Linear(self.d_model, self.d_model) 
        self.inference_layer = nn.Linear(self.d_model, self.d_model)
        self.mu_z_t_hat_layer = nn.Linear(self.d_model, self.d_model)
        self.var_z_t_hat_layer = nn.Linear(self.d_model, self.d_model)
        self.mixing_model = mixing_model

    
    def get_coords(self, L, H):
        coords = torch.linspace(0, 1, L+H)
        # reshape
        return rearrange(coords, 't -> 1 t 1')
    
    
    def reparameterization(self, mu, var):
        return (torch.randn_like(torch.exp(0.5 * var)) * torch.exp(0.5 * var) + mu)


    def forward(self, batch_x, time_x, dec_inp, is_eval):
        if self.temporal_index is True:
            coords = self.get_coords(self.L, self.H).to(self.device)
            time_x = repeat(self.CoordEmbedding(coords), '1 t d -> b t d', b=time_x.size(0))
            time_x = self.map_layer(time_x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            time_x = self.TimeFeatureEmbedding(time_x)
        
        # VAE [B, N, D] -> [B, N, D']
        mu_t = self.mu_t_layer(time_x)
        var_t = self.var_t_layer(time_x)
        z_t = self.reparameterization(mu_t, var_t)
        
        enc_out, attns, seasonal_init, trend_init = self.mixing_model.Encoder(batch_x, None, dec_inp, None)
        enc_out = self.gate_layer_1(time_x) * enc_out 

        # Dynamic Inference
        z_t_hat = self.inference_layer(enc_out)
        mu_z_t_hat = self.mu_z_t_hat_layer(z_t_hat)
        var_z_t_hat  = self.var_z_t_hat_layer(z_t_hat)

        mu_z = self.mu_z_layer(enc_out)
        var_z = self.var_z_layer(enc_out)
        z = self.reparameterization(mu_z, var_z)

        forecast_y = self.mixing_model.Decoder(z, attns, seasonal_init, trend_init, batch_x, None, dec_inp, None)
        return forecast_y, mu_t, var_t, mu_z_t_hat, var_z_t_hat, mu_z, var_z