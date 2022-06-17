from PIL import Image
from pathlib import Path
import numpy as np
import re
import time

import pandas as pd

new_data_path = Path("/mnt/ssd1/FIL_dataset_mmwave")
# processed_new_data = Path("/dev/null")
processed_new_data = Path("/home/yuanjie/mixed")
mid_depth = Path("/home/yuanjie/depth_mid")


pt = re.compile(r'(?<=a)[0-9]{2}|(?<=s)[0-9]{2}')

def split_train_test(inpath:Path, outpath:Path):
    hz = 30
    timestep = 0.5
    window = 2

    data_count = 0
    train_subj = list(np.arange(2, 10))
    test_subj = list(np.arange(1, 2))

    # train_path = processed_new_data.joinpath("train")
    # test_path = processed_new_data.joinpath("test")

    # train_path.mkdir(parents=True)
    # test_path.mkdir(parents=True)
    folder_list = sorted(inpath.glob("*"), key=lambda x: int(pt.findall(x.name)[0]))
    for fd in folder_list:#inpath.glob("*"):
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
            depth_data = np.load(mid_depth.joinpath(fd.name, "depth.npz"))["arr_0"]
            print(mid_depth.joinpath(fd.name, "depth.npz"))
            print(raw_data.shape)
            print(depth_data.shape)

            smaller = min(raw_data.shape[0], depth_data.shape[0])
            for i in range(smaller):
                one_data = np.expand_dims(raw_data[i], 1)
                one_depth = depth_data[i]
                labels = np.asarray([act])
                # print(one_data.shape)
                # print(one_depth.shape)

                # time.sleep(10)
                # if i < 0.8 * raw_data.shape[0]:savepath = outpath.joinpath(f"train/{data_count}.npz")
                # elif i >= 0.8 * raw_data.shape[0]:savepath = outpath.joinpath(f"test/{data_count}.npz")
                if subj < 10: 
                    savepath = outpath.joinpath(f"train/{data_count}_mm.npz")
                    savepath2 = outpath.joinpath(f"train/{data_count}_depth.npz")
                elif subj == 10: 
                    savepath = outpath.joinpath(f"test/{data_count}_mm.npz")
                    savepath2 = outpath.joinpath(f"test/{data_count}_depth.npz")

                data_count += 1
                np.savez(savepath, one_data, labels)
                np.savez(savepath2, one_depth, labels)

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
# generate_list(processed_new_data)