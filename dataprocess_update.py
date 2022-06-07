from pathlib import Path
import time
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as tf

from util.getlog import get_log
logger = get_log()

class Eat_data(Dataset):
    def __init__(self, data_path_list, selection:str, split:str) -> None:
        super(Eat_data, self).__init__()

        assert split in ["train", "test"]

        self.path = Path(data_path_list[selection])
        self.selection = selection
        self.split = split
        self.name_list = self.get_data()



    def get_data(self):
        path = self.path
        data_list = path.joinpath(f"{self.split}.txt")
        with open(data_list, "r") as f:
            return [name.strip() for name in f.readlines()]

    def __getitem__(self, index):
        data_path = self.path.joinpath(self.split)
        fl = np.load(data_path.joinpath(f"{self.name_list[index]}.npz"))
        data = torch.tensor(fl['arr_0'], dtype=torch.float32)
        label = torch.tensor(fl['arr_1'][0], dtype=torch.long)
        transdict = {
            0:0,
            1:0,
            2:0,
            3:1,
            4:1,
            5:1,
            6:2,
            7:2,
            8:2,
            9:2,
            10:2,
            11:2,
        }
        label = torch.tensor(transdict[int(label)], dtype=torch.long)
        if self.selection == "DEPTH_LSTM" and self.split == "train":
            p1, p2, h, w = transforms.RandomCrop.get_params(torch.randn((1, 240, 240)), (228, 228))
            resize = transforms.Resize((240, 240), transforms.InterpolationMode.BILINEAR)
            croped = []
            i = 0
            #print(self.data[index][i].size())
            while i < (data.size()[0]):
                croped.append(resize(tf.crop(torch.unsqueeze(data[i], dim=0), p1, p2, h, w)))
                i = i + 1

            croped = torch.cat(croped, dim=0)
            return croped, label
        # elif self.selection == "MARS":
        #     img = data.reshape((3, 3, 11))
        #     return img, label
        return data, label

    def __len__(self):
        return len(self.name_list)