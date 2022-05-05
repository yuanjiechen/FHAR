from pathlib import Path
import time
import torch
from torch.utils.data import Dataset
import numpy as np

from util.getlog import get_log
logger = get_log()

class Eat_data(Dataset):
    def __init__(self, data_path_list, selection:str, split:str) -> None:
        super(Eat_data, self).__init__()

        assert split in ["train", "test"]

        self.path_list = data_path_list
        self.path = Path(self.path_list[selection]).joinpath(split)

        try:
            self.data, self.label = eval(f"self.{selection}_process")(self.path)
        except BaseException:
            raise NotImplementedError("No such dataset process method !")

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.label)

    def CNN_LSTM_process(self, path):
        data = []
        label = []
        # transdict = {
        #     'a01':0,
        #     'a02':1,
        #     'a03':2,
        #     'a04':3,
        #     'a05':4,
        #     'a06':5,
        #     'a07':6,
        #     'a08':7,
        #     'a09':8,
        #     'a10':9,
        #     'a11':10,
        #     'a12':11,
        # }
        # transdict = {
        #     'a01':0,
        #     'a02':1,
        #     'a04':2,
        #     'a06':3,
        #     'a08':4,
        #     'a10':5,
        #     'a11':6,
        #     'a12':7,
        # }
        transdict = {
            'a01':0,
            'a02':0,
            'a03':0,
            'a04':1,
            'a05':1,
            'a06':1,
            'a07':2,
            'a08':2,
            'a09':2,
            'a10':2,
            'a11':2,
            'a12':2,
        }
        # transdict = {
        #     'boxing':0,
        #     'squats':1,
        #     'jack':2,
        #     'jump':3,
        #     'walk':4,
        # }

        for file in path.rglob("*.npz"):
            f_data = np.load(file)

            data.append(f_data['arr_0'])
            tp_label = np.asarray([transdict[ele] for ele in f_data['arr_1']])
            label.append(tp_label)

        data = np.concatenate(data, axis=0)
        data = np.expand_dims(data, axis=2)
        label = np.concatenate(label, axis=0)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def CNN_LN_process(self, path):
        data = []
        label = []

        for file in path.rglob("*.npz"):
            f_data = np.load(file)

            data.append(f_data['arr_0'])
            tp_label = np.asarray(f_data['arr_1'])
            label.append(tp_label)

        data = np.concatenate(data, axis=0)
        label = np.concatenate(label, axis=0)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)        

    def Fused_process(self, _):
        path_1 = Path(self.path_list["CNN_LSTM"])
        path_2 = Path(self.path_list["CNN_LN"])

        data_1, label = self.CNN_LSTM_process(path_1)
        data_2, _ = self.CNN_LN_process(path_2)

        return [data_1, data_2], label

    def CNN_DEPTH_process(self):
        pass
# ddd = Eat_data(Path("../Processed_Data/"), "CNN_LSTM", "train")