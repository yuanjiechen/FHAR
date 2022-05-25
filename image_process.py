from PIL import Image
from pathlib import Path
import numpy as np
import re
import time

import pandas as pd


depth_path = Path("/home/deepblue/mmWave_parse/depth_images")
processed_depth = Path("/home/yuanjie/depth_")

skeleton_path = Path("/home/deepblue/mmWave_parse/skeleton_images")
processed_skeleton = Path("/home/yuanjie/skeloton")

pt = re.compile(r'(?<=a)[0-9]{2}|(?<=s)[0-9]{2}')

def split_train_test(type="depth"):
    hz = 30
    timestep = 0.5
    window = 2

    dic = {0:0, 3:1, 6:2, 7:3, 8:4, 9:5, 10:6, 11:7, 12:8}
    data_count = 0
    for fd in skeleton_path.glob("*"):
        if fd.name in ['train', "test"]:
            continue
        else:
            print(fd.name)
            try:
                act, subj = pt.findall(fd.name)
            except:
                continue
            act = int(act) - 1#dic[int(act) - 1]
            subj = int(subj)

            if type == "mmwave": img_list = sorted(fd.glob("*.npz"), key=lambda x:int(x.stem))
            elif type == "depth": img_list = sorted(fd.glob("*.png"), key=lambda x:int(x.stem))
            elif type == "skeloton": 
                img_list = list(fd.rglob("*.csv"))
                data = pd.read_csv(img_list[0], header=None)
                i = 0
                while i < len(list(data.index)):
                    one_data = np.asarray(data.iloc[i, :], dtype=np.float32)
                    labels = np.asarray([act])
                    i += 1
                    if subj < 10:
                        savepath = processed_skeleton.joinpath(f"train/{data_count}.npz")

                    else:
                        savepath = processed_skeleton.joinpath(f"test/{data_count}.npz")
                    
                    data_count += 1
                    # print(one_data[0])
                    np.savez(savepath, one_data, labels)

            if type == "mmwave" or type == "depth":
                i = 0
                one_data = []
                while (i + hz * window) < len(img_list):
                    sample = []
                    tmp_list = img_list[i:i + hz * window]  
                    for image in tmp_list:
                        if type == "mmwave": img = np.load(image)["arr_0"]
                        elif type == "depth": img = np.asarray(Image.open(image, "r"))

                        sample.append(img)
                    
                    one_data = np.stack(sample, axis=0)
                    labels = np.asarray([act])
                    # one_data.append(np.stack(sample, axis=0))
                    i += int(timestep * hz)

                    if subj < 10:
                        savepath = processed_skeleton.joinpath(f"train/{data_count}.npz")

                    else:
                        savepath = processed_skeleton.joinpath(f"test/{data_count}.npz")
                    
                    data_count += 1
                    print(one_data.shape)
                    np.savez(savepath, one_data, labels)


def generate_list():
    train_path = processed_skeleton.joinpath("train/")
    test_path = processed_skeleton.joinpath("test/")

    train_list = [name.stem for name in train_path.glob("*.npz")]
    test_list = [name.stem for name in test_path.glob("*.npz")]

    with open(processed_skeleton.joinpath("train.txt"), "w+") as f:
        for name in train_list:
            f.write(f"{name}\n")
    
    with open(processed_skeleton.joinpath("test.txt"), "w+") as f:
        for name in test_list:
            f.write(f"{name}\n")
        

split_train_test("skeloton")
generate_list()