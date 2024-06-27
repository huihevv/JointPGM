import torch
import numpy as np

def setup_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_model_parameters(model, only_num=True):
    print('**************Model Parameter**************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: %.2fM' % (total_num / 1e6))
    print('**************Finish Parameter*************')

def get_init_batch(batch, device, series_mixing, pred_len, label_len):
    batch_x, batch_y, time_x, time_y = batch
    batch_x, batch_y, time_x, time_y = batch_x.to(device).float(), batch_y.to(device).float(), \
                                    time_x.to(device).float(), time_y.to(device).float()
    if series_mixing is False:
        return batch_x, batch_y, time_x, time_y
    else:
        dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :])
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1)
        return batch_x,  batch_y, time_x, time_y, dec_inp