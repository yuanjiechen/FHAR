from pathlib import Path
from pickle import LIST
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def get_loss(selection):
    loss_dict = {
        "CNN_LSTM":nn.CrossEntropyLoss(), 
        "CNN_LN": nn.CrossEntropyLoss(),
        "CNN_DEPTH":nn.CrossEntropyLoss(),
        "DEPTH_LSTM":nn.CrossEntropyLoss(),
        "MARS":nn.CrossEntropyLoss()
    }
    if selection in list(loss_dict.keys()): return loss_dict[selection]
    else: return nn.CrossEntropyLoss()

def get_optimizer(selection, params, lr, decay=0):

    # optim_dict = {
    #     "CNN_LSTM":torch.optim.Adam(params, lr, weight_decay=decay),
    #     #"CNN_LN":torch.optim.Adam(params, lr, weight_decay=decay)
    # }
    if selection == "MARS": return torch.optim.Adam(params, lr, weight_decay=decay, betas=[0.5, 0.999], amsgrad=False)
    return torch.optim.Adam(params, lr, weight_decay=decay)

def get_lr_decay(selection, optim, decay):
    if decay == 0:
        return None
        
    decay_dict = {
        "CNN_LSTM":torch.optim.lr_scheduler.ExponentialLR(optim, decay),
        "CNN_LN": torch.optim.lr_scheduler.ExponentialLR(optim, decay),
        "CNN_DEPTH": torch.optim.lr_scheduler.ExponentialLR(optim, decay),
        "DEPTH_LSTM": torch.optim.lr_scheduler.ExponentialLR(optim, decay),
        "MARS": torch.optim.lr_scheduler.ExponentialLR(optim, decay)
    }

    if selection in list(decay_dict.keys()): return decay_dict[selection]
    else: return torch.optim.lr_scheduler.ExponentialLR(optim, decay)

def cal_result(labels:torch.Tensor, predictions:torch.Tensor, n_classes:int, average=None):
    classes = list(np.arange(n_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), predictions.cpu(), labels=classes, average=average)

    return precision, recall, f1

def calculate_accuracy(logits, labels):
    # inputs should be torch.tensor
    predictions = logits.argmax(1)
    no_count = (labels==-1).sum()
    count = ((predictions==labels)*(labels!=-1)).sum()
    acc = count.float() / (labels.numel()-no_count).float()
    return acc



def get_config(selection, cfgpath=Path("./train_cfg.csv")):
    '''
        data_path [0]
        learning_rate [1]
        batch_size [2]
        epoch [3]
        weight_decay [4]
        lr_decay [5]
    '''
    configs = pd.read_csv(cfgpath, index_col=0)
    cfg = configs.loc[selection, :]

    return dict(zip(configs.index, np.reshape(configs["data_path"].values, -1))), float(cfg[1]), int(cfg[2]), int(cfg[3]), float(cfg[4]), float(cfg[5])

def calculate_result(cf):
    n_class = cf.shape[0]
    conf = np.zeros((n_class,n_class))
    class_acc = np.zeros(n_class)
    IoU = np.zeros(n_class)

    conf[:,0] = cf[:,0]/cf[:,0].sum()

    for pred in range(0, n_class):
        class_acc[pred] = cf[pred,pred] / cf[pred, :].sum()

    for cid in range(1,n_class):
        conf[:,cid] = cf[:,cid]/cf[:,cid].sum()
        IoU[cid]  = cf[cid,cid]/(cf[cid,1:].sum()+cf[1:,cid].sum()-cf[cid,cid])
    overall_acc = np.diag(cf[1:,1:]).sum()/cf[1:,:].sum()
    acc = np.diag(conf)

    return overall_acc, acc, IoU, class_acc