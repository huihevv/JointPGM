import time
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from utils import setup_seed, print_model_parameters, get_init_batch
from utils.param_parser import parameter_parser, update_args_from_model_params
from utils.dataset import TSForecastDataset
from utils.early_stopping import EarlyStopping
from utils.metrics import get_metrics
from backbones import Autoformer, Informer, Transformer
from model.DSDMTS import DSDMTS
from model.DSMMTS import DSMMTS

torch.autograd.set_detect_anomaly(True)
args = parameter_parser()

# set seed
setup_seed(args.seed)
# select device
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

# prepare dataset
val_ratio, test_ratio = 0.1, 0.2
data_path = 'dataset/{}.csv'.format(args.data)

train_dataset = TSForecastDataset(data_path, flag='train', size=(args.seq_len, args.pred_len), split=(val_ratio, test_ratio), scale=args.scale, 
                                  scale_method=args.scale_method, time_features=args.time_features, normalise_time_features=args.normalise_time_features)
val_dataset = TSForecastDataset(data_path, flag='val', size=(args.seq_len, args.pred_len), split=(val_ratio, test_ratio), scale=args.scale, 
                                scale_method=args.scale_method, time_features=args.time_features, normalise_time_features=args.normalise_time_features)
test_dataset = TSForecastDataset(data_path, flag='test', size=(args.seq_len, args.pred_len), split=(val_ratio, test_ratio), scale=args.scale, 
                                 scale_method=args.scale_method, time_features=args.time_features, normalise_time_features=args.normalise_time_features)

# set forecast dataloader
train_loader  = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)


# initialize model
N_series = train_dataset.N
args = update_args_from_model_params(args, N_series, device)

# DSD-MTS or DSM-MTS
if args.series_mixing is False:
    model = DSDMTS(args).to(device)
else:
    # set backbones like Dish-TS
    MODEL = args.mixing_model
    model_dict = {'Autoformer': Autoformer, 'Transformer': Transformer, 'Informer': Informer}
    mixing_model = model_dict[MODEL].Model(args)
    model = DSMMTS(args, mixing_model).to(device)


optimizer = optim.Adam(model.parameters(), lr=args.lr)
early_stopping = EarlyStopping(patience=args.patience, verbose=True, dump=False)
loss_fore = nn.MSELoss()
loss_reco = nn.MSELoss()

# print parameter volume
print_model_parameters(model, only_num=True)

max_epochs = 100
for epoch in range(max_epochs):
    train_losses, val_losses = [], []
    # train
    begin_time = time.time()
    model.train() 
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        if args.series_mixing is False:
            batch_x, batch_y, time_x, time_y = get_init_batch(batch, device, args.series_mixing, args.pred_len, args.label_len)
            forecast_x, forecast_y, mu_t, var_t, mu_z_t_hat, var_z_t_hat, mu_z, var_z = model(batch_x, time_x, is_eval=False)
            loss_r = loss_reco(forecast_x, batch_x)
        else: 
            batch_x, batch_y, time_x, time_y, dec_inp = get_init_batch(batch, device, args.series_mixing, args.pred_len, args.label_len)
            forecast_y, mu_t, var_t, mu_z_t_hat, var_z_t_hat, mu_z, var_z = model(batch_x, time_x, dec_inp, is_eval=False)

        loss_f = loss_fore(forecast_y, batch_y)
        loss_kl1 = -0.5*torch.sum(var_z + 1 - mu_z.pow(2) - var_z.exp())
        loss_kl2 = -0.5*torch.sum((var_z_t_hat-var_t) + 1 - (mu_z_t_hat - mu_t).pow(2)/var_t.exp() - (var_z_t_hat).exp()/var_t.exp())
        if args.series_mixing is False:
            loss = loss_r + loss_f + 0.1 * loss_kl1 + 0.1 * loss_kl2
        else:
            loss = loss_f + loss_kl1 + loss_kl2
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    end_time = time.time()

    # validate
    with torch.no_grad():
        model.eval()
        for batch in val_loader:
            if args.series_mixing is False:
                batch_x, batch_y, time_x, time_y = get_init_batch(batch, device, args.series_mixing, args.pred_len, args.label_len)
                forecast_x, forecast_y, mu_t, var_t, mu_z_t_hat, var_z_t_hat, mu_z, var_z = model(batch_x, time_x, is_eval=True)
                loss_r = loss_reco(forecast_x, batch_x)
            else:
                batch_x, batch_y, time_x, time_y, dec_inp = get_init_batch(batch, device, args.series_mixing, args.pred_len, args.label_len)
                forecast_y, mu_t, var_t, mu_z_t_hat, var_z_t_hat, mu_z, var_z = model(batch_x, time_x, dec_inp, is_eval=False)
            loss_f = loss_fore(forecast_y, batch_y)
            loss_kl1 = -0.5*torch.sum(var_z + 1 - mu_z.pow(2) - var_z.exp())
            loss_kl2 = -0.5*torch.sum((var_z_t_hat-var_t) + 1 - (mu_z_t_hat - mu_t).pow(2)/var_t.exp() - (var_z_t_hat).exp()/var_t.exp())
            if args.series_mixing is False:
                loss = loss_r + loss_f + 0.1 * loss_kl1 + 0.1 * loss_kl2
            else:
                loss = loss_f + loss_kl1 + loss_kl2
            val_losses.append(loss.item())
    # early stop
    print('epoch:{0:}, training time: {1:.5f}s, train_loss:{2:.5f}, val_loss:{3:.5f}'.format(epoch, (end_time-begin_time), 
        np.mean(train_losses), np.mean(val_losses)))
    early_stopping(np.mean(val_losses), model, epoch)
    if early_stopping.early_stop:
        print("Early stopping with best_score:{}".format(-early_stopping.best_score))
        break
    if np.isnan(np.mean(val_losses)) or np.isnan(np.mean(train_losses)):
        break

# test
begin_time = time.time()
model = early_stopping.best_model
model.eval()
preds, trues = None, None
with torch.no_grad():
    for batch in test_loader:
        if args.series_mixing is False:
            batch_x, batch_y, time_x, time_y = get_init_batch(batch, device, args.series_mixing, args.pred_len, args.label_len)
            forecast_x, forecast_y, mu_t, var_t, mu_z_t_hat, var_z_t_hat, mu_z, var_z = model(batch_x, time_x, is_eval=True)
        else:
            batch_x, batch_y, time_x, time_y, dec_inp = get_init_batch(batch, device, args.series_mixing, args.pred_len, args.label_len)
            forecast_y, mu_t, var_t, mu_z_t_hat, var_z_t_hat, mu_z, var_z = model(batch_x, time_x, dec_inp, is_eval=False)
        # concat
        pred = forecast_y.detach().cpu().numpy()
        true = batch_y.detach().cpu().numpy()

        preds = pred if preds is None else np.concatenate((preds, pred), 0)
        trues = true if trues is None else np.concatenate((trues, true), 0)
preds_inverse, trues_inverse = None, None
if args.do_predict:
    for i in range(preds.shape[0]):
        pred_inverse = np.expand_dims(test_dataset.inverse_transform(preds[i]), axis=0)
        true_inverse = np.expand_dims(test_dataset.inverse_transform(trues[i]), axis=0)
        preds_inverse = pred_inverse if preds_inverse is None else np.concatenate((preds_inverse, pred_inverse), 0)
        trues_inverse = true_inverse if trues_inverse is None else np.concatenate((trues_inverse, true_inverse), 0)
    mae, mse, rmse, mape, smape, wape, mspe, var = get_metrics(preds_inverse, trues_inverse)     
else:
    mae, mse, rmse, mape, smape, wape, mspe, var = get_metrics(preds, trues)
print('Performance on test set: MAE: {:5.4f}, MSE: {:5.4f}, RMSE: {:5.4f}, MAPE: {:5.4f}, SMAPE: {:5.4f}, WAPE: {:5.4f}' 
      ', MSPE: {:5.4f}, VAR: {:5.4f}'.format(mae, mse, rmse, mape, smape, wape, mspe, var))
print('Evaluation took time: {:5.2f}s.'.format((time.time() - begin_time)))
print('done')




