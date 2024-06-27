import argparse

def update_args_from_model_params(args, N_series, device):
    model_params = {
        'N_series': N_series, 'device': device, # JointPGM
        "embed_type":3,'factor':3,"output_attention":False,'d_model':512,'embed':'timeF','freq':'h', # Informer
        'dropout':0.05, 'n_heads':8,'d_ff':2048, 'moving_avg':25,'activation':'gelu','e_layers':2, # Autoformer
        'version':'Fourier', 'mode_select': 'random', 'modes': 64, 'L': 3, 'base': 'legendre', 'cross_activation': 'tanh', # FEDformer
        'use_norm': True, 'class_strategy': 'projection', # iTransformer
        'd_layers':1, 'distil':True, "enc_in":N_series, "dec_in":N_series, 'c_out':N_series,
        }
    model_params.update(vars(args))
    args = argparse.Namespace(**model_params)
    return args

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run non-stationary MTS forecasting task.")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--output_file', type=str, default='forecast.csv')
    # forecast
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=192)
    parser.add_argument('--temporal_index', type=bool, default=True, help='temporal index or timestamp features?')
    parser.add_argument('--individual', type=bool, default=False, help='parameter sharing or not?')
    # data
    parser.add_argument('--data', type=str, default='ETTh1') 
    parser.add_argument('--time_features', type=str, default='s', help='Freq for time features encoding')
    parser.add_argument('--normalise_time_features', type=bool, default=True)
    parser.add_argument('--scale', type=bool, default=True)
    parser.add_argument('--scale_method', type=str, default='std', choices=['std', 'min-max'], help="Normalization method")
    # forecast model
    parser.add_argument('--fourier_feat_size', type=int, default=4096, help='Fourier Feature Size')
    parser.add_argument('--n_coords_layers', type=int, default=1, help='Coordinate Embedding Layers')
    parser.add_argument('--sigma_s', type=list, default=[0.01, 0.1, 1, 5, 10, 20, 50, 100], help='Scale of B')
    parser.add_argument('--temp', type=float, default=0.5, help='Temperature value for gumbel-softmax')
    parser.add_argument('--gc_deep', type=int, default=2, help='Depth of Graph Convolution')
    parser.add_argument('--belta', type=float, default=0.1, help='Weight of KL loss')
    parser.add_argument('--alpha', type=float, default=0.5, help='Trade-off parameter')
    # jointpgm or jointpgm_mixing?
    parser.add_argument('--series_mixing', type=bool, default=False, help='series mixing or independent?')  
    # set jointpgm_mixing backbones
    parser.add_argument('--mixing_model', type=str, default='Autoformer')  
    # embed_size and latent_size are generally consistent, d in paper
    parser.add_argument('--embed_size', type=int, default=128) 
    parser.add_argument('--latent_size', type=int, default=128) 
    parser.add_argument('--do_predict', type=bool, default=False, help='whether to predict unseen future data')
    # optimization
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--gpu', type=int, default=1)
    return parser.parse_args()