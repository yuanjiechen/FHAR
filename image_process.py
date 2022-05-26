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

mm_path = Path("/home/deepblue/mmWave_parse/mmwave_points")
processed_mm = Path("/home/yuanjie/mm")

pt = re.compile(r'(?<=a)[0-9]{2}|(?<=s)[0-9]{2}')

def split_train_test(type="depth"):
    hz = 30
    timestep = 0.5
    window = 2

    dic = {0:0, 3:1, 6:2, 7:3, 8:4, 9:5, 10:6, 11:7, 12:8}
    data_count = 0
    for fd in mm_path.glob("*"):
        if fd.name in ['train', "test"]:
            continue
        else:
            print(fd.name)
            try:
                act, subj = pt.findall(fd.name)
            except:
                continue
            act = int(act) - 1
            subj = int(subj)


            if type == "skeloton": 
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

            elif type == "mm":
                img_list = list(fd.rglob("*.npy"))
                print(fd)
                raw_data = np.load(img_list[0], allow_pickle=True).item()
                key = list(raw_data.keys())
                i = 0

                while (i + hz * window) < len(key):
                    sample = []
                    fetch_key = key[i:i + hz * window]
                    for k in fetch_key:
                        one_frame = np.asarray(raw_data[k], dtype=np.float32)
                        #print(one_frame.shape)
                        frame64 = np.zeros((64, 5), dtype=np.float32)
                        if one_frame.shape[0] != 0:
                            frame64[:one_frame.shape[0], :one_frame.shape[1]] = one_frame
                            frame64 = frame64[np.lexsort(keys=(frame64[:, 4], frame64[:, 3], frame64[:, 2], frame64[:, 1], frame64[:, 0]))]
                        sample.append(frame64)
                    sample = np.stack(sample, axis=0)
                    sample = sample.transpose(0, 2, 1).reshape((60, 5, 8, 8))
                    #print(sample.shape)

                    labels = np.asarray([act])
                    i += int(timestep * hz)
                    if subj < 10:
                        savepath = processed_mm.joinpath(f"train/{data_count}.npz")

                    else:
                        savepath = processed_mm.joinpath(f"test/{data_count}.npz")
                    
                    data_count += 1

                    np.savez(savepath, sample, labels)

            elif type == "mmwave" or type == "depth":
                if type == "mmwave": img_list = sorted(fd.glob("*.npz"), key=lambda x:int(x.stem))
                elif type == "depth": img_list = sorted(fd.glob("*.png"), key=lambda x:int(x.stem))
                i = 0

                while (i + hz * window) < len(img_list):
                    sample = []
                    tmp_list = img_list[i:i + hz * window]  
                    for image in tmp_list:
                        if type == "mmwave": img = np.load(image)["arr_0"]
                        elif type == "depth": img = np.asarray(Image.open(image, "r"))

                        sample.append(img)
                    
                    sample = np.stack(sample, axis=0)
                    labels = np.asarray([act])
                    i += int(timestep * hz)

                    if subj < 10:
                        savepath = processed_skeleton.joinpath(f"train/{data_count}.npz")

                    else:
                        savepath = processed_skeleton.joinpath(f"test/{data_count}.npz")
                    
                    data_count += 1

                    np.savez(savepath, sample, labels)


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
        

split_train_test("mm")
generate_list(processed_mm)