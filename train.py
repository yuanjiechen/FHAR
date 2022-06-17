from datetime import datetime
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np

from util.getlog import get_log
from util.dev_check import gpu_check
from train_util import get_loss, get_lr_decay, get_optimizer, cal_result, get_config, calculate_accuracy, calculate_result
from dataprocess_update import Eat_data
from models import Model
logger = get_log()
torch.manual_seed(7)
class Train():
    def __init__(self, args) -> None:
        self.args = args
        self.data_path_list, self.lr, self.batch_sz, self.epoch, self.weight_decay, self.lr_decay = get_config(self.args.selection)

        self.device = self.init_device()
        self.train_loader, self.test_loader = self.init_data()
        


    def init_device(self):
        if self.args.gpu == -1:
            return "cpu"

        elif gpu_check(self.args.gpu):
            return f"cuda:{self.args.gpu}"

        else:
            raise f"{[datetime.now()]}Selected GPU device is unavailable !"

    def init_data(self):
        train_set = Eat_data(
            data_path_list=self.data_path_list,
            selection=self.args.selection,
            split="train"
        )

        test_set = Eat_data(
            data_path_list=self.data_path_list,
            selection=self.args.selection,
            split="test"
        )

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.batch_sz,
            shuffle=True,
            num_workers=10,
            drop_last=True
        )

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.batch_sz,
            shuffle=False,
            num_workers=10,
            drop_last=True
        )

        return train_loader, test_loader

    def trainer(self, path:Path):

        n_classes = 12
        titles = ["train_acc", "acc", "train_loss", "test_loss"].extend([f"class{i}_recall" for i in range(n_classes)].extend([f"class{i}_precision" for i in range(n_classes)])) 
        
        model = Model(n_classes, self.args.selection).to(self.device)
        loss_func = get_loss(self.args.selection)
        align_loss = nn.MSELoss()
        opt = get_optimizer(self.args.selection, model.parameters(), self.lr, self.weight_decay)
        lr_decayer = get_lr_decay(self.args.selection, opt, self.lr_decay)

        lib = Path(str(path)[:-4]+'/')
        lib.mkdir(parents=True, exist_ok=True)
        
        for i in range(1, self.epoch):
            cf = np.array((n_classes, n_classes))
            result = []
            model.train()
            train_loss = 0.0
            total_train_acc = 0.0
            train_start_time = time.time()*1000
            for it, (data, labels) in enumerate(self.train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)

                opt.zero_grad()
                pred, kd_ele = model(data)


                train_acc = calculate_accuracy(pred, labels)
                total_train_acc += train_acc.item()
                loss = loss_func(pred, labels)

                if self.args.selection == "Fused":
                    loss_2 = align_loss(kd_ele[0], kd_ele[1])
                    
                train_loss += loss.item()
                loss.backward()
                opt.step()
            train_end_time = time.time()*1000
            train_time = train_end_time-train_start_time
            
            pred_list = []
            label_list = []
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                index = 0
                total_acc = 0.0
                infer_time = 0.0
                for it, (data, labels) in enumerate(self.test_loader):
                    index+=1
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    start_time = time.time()*1000
                    pred, _ = model(data)
                    end_time = time.time()*1000
                    infer_time+=(end_time-start_time)
                    loss = loss_func(pred, labels)
                    test_loss += loss.item()
                    acc = calculate_accuracy(pred, labels)
                    total_acc+= acc.item()
                    pred = pred.argmax(1)

                    pred_list.append(pred.cpu())
                    label_list.append(labels.cpu())
                average_inf_time = infer_time/(len(self.train_loader)*10)
          
                    
                pred_ = torch.concat(pred_list, dim=0)
                label_ = torch.concat(label_list, dim=0)
                precision, recall, f1 = cal_result(label_, pred_, n_classes)

                total_acc/=index
                print(f"[{datetime.now()}] Epoch {i} evaluation report: ")
                print('| Training accuracy', total_train_acc / len(self.train_loader))
                print('| Test accuracy:', total_acc)
                print('| Train loss', train_loss / len(self.train_loader))
                print('| Precision:', precision)
                print('| recall', recall)
                print('| loss:', test_loss / len(self.test_loader))
                print('| Train time:', train_time)
                print('| Average time:', average_inf_time)


            if lr_decayer is not None: lr_decayer.step()
            res = [total_train_acc / len(self.train_loader), total_acc, train_loss / len(self.train_loader), test_loss / len(self.test_loader)]
            res.extend(recall)
            res.extend(precision)
            result.append(res)
            df = DataFrame(result, columns=titles, index=[i])
            if i == 1:
                df.to_csv(str(path), mode="a+")
            else:
                df.to_csv(str(path), mode="a+", header=None)
            
            tmp = Path(str(path)[:-4]+'/'+str(i))
            torch.save(model.state_dict(), str(tmp.with_suffix(".pth")))