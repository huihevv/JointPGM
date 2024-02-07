import numpy as np

# metrics
def MAE(preds, trues):
    return np.mean(np.abs(preds - trues))

def MSE(preds, trues):
    return np.mean((preds - trues)**2)

def RMSE(preds, trues):
    return np.sqrt(np.mean((preds - trues)**2))

def MAPE(preds, trues):
    trues = trues + 1e-8
    return np.mean(np.abs((preds - trues) / trues))

def SMAPE(preds, trues):
    trues = trues + 1e-8
    return np.mean((2*np.abs(preds - trues) / (np.abs(preds) + np.abs(trues))))

def WAPE(preds, trues):
    trues = trues + 1e-8
    return np.mean(np.abs(preds - trues)) / np.mean(np.abs(trues))

def MSPE(preds, trues):
    trues = trues + 1e-8
    return np.mean(np.square((preds - trues) / trues))

def VAR(preds, trues):
    return np.var(np.abs(preds - trues))


def get_metrics(preds, trues):
    mae = MAE(preds, trues)
    mse = MSE(preds, trues)
    rmse = RMSE(preds, trues)
    mape = MAPE(preds, trues)
    smape = SMAPE(preds, trues)
    wape = WAPE(preds, trues)
    mspe = MSPE(preds, trues)
    var = VAR(preds, trues)
    return mae, mse, rmse, mape, smape, wape, mspe, var