from PIL import Image
from pathlib import Path
import numpy as np
import re
import time

import pandas as pd

new_data_path = Path("/mnt/ssd1/new_food_dataset/")
processed_new_data = Path("/mnt/ssd1/processed_dataset")

pt = re.compile(r'(?<=a)[0-9]{2}|(?<=s)[0-9]{2}')

def split_train_test(inpath:Path, outpath:Path):
    hz = 30
    timestep = 0.5
    window = 2

    data_count = 0
    train_subj = list(np.arange(3, 10))
    test_subj = list(np.arange(1, 3))

    train_path = processed_new_data.joinpath("train")
    test_path = processed_new_data.joinpath("test")

    train_path.mkdir(parents=True)
    test_path.mkdir(parents=True)

    for fd in inpath.glob("*"):
        print(fd.name)
        try:
            act, subj = pt.findall(fd.name)
        except:
            continue
        act = int(act) - 1
        subj = int(subj)

        if True:
            data_list = list(fd.rglob("*.npz"))
            raw_data = np.load(data_list[0])["arr_0"]
            for i in range(raw_data.shape[0]):
                one_data = np.expand_dims(raw_data[i], 1)
                labels = np.asarray([act])

                if subj in train_subj: savepath = outpath.joinpath(f"train/{data_count}.npz")
                elif subj in test_subj: savepath = outpath.joinpath(f"test/{data_count}.npz")
                data_count += 1
                np.savez(savepath, one_data, labels)

def generate_list(path:Path):
    train_path = path.joinpath("train/")
    test_path = path.joinpath("test/")

    train_list = [name.stem for name in train_path.glob("*.npz")]
    test_list = [name.stem for name in test_path.glob("*.npz")]

    with open(path.joinpath("train.txt"), "w+") as f:
        for name in train_list:
            f.write(f"{name}\n")
    
    with open(path.joinpath("test.txt"), "w+") as f:
        for name in test_list:
            f.write(f"{name}\n")
        

split_train_test(new_data_path, processed_new_data)
generate_list(processed_new_data)