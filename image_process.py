from PIL import Image
from pathlib import Path
import numpy as np
import re



depth_path = Path("/home/deepblue/mmWave_parse/depth_images")
processed_depth = Path("/home/yuanjie/depth")
pt = re.compile(r'(?<=a)[0-9]{2}|(?<=s)[0-9]{2}')

def split_train_test(type="depth"):
    hz = 30
    timestep = 0.5
    window = 2

    dic = {0:0, 3:1, 6:2, 7:3, 8:4, 9:5, 10:6, 11:7, 12:8}
    data_count = 0
    for fd in depth_path.glob("*"):
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
            i = 0
            one_data = []
            while (i + hz * window) < len(img_list):
                sample = []
                tmp_list = img_list[i:i + hz * window]  
                for image in tmp_list:
                    if type == "mmwave": img = np.load(image)["arr_0"]
                    elif type == "depth": img = np.asarray(Image.open(image, "r"))

                    sample.append(img)
                
                one_data.append(np.stack(sample, axis=0))
                i += int(timestep * hz)
            
            train_data = np.stack(one_data, axis=0)
            labels = np.repeat(act, train_data.shape[0])

            if subj < 10:
                savepath = processed_depth.joinpath(f"train/{data_count}.npz")

            else:
                savepath = processed_depth.joinpath(f"test/{data_count}.npz")
            
            data_count += 1
            print(train_data.shape)
            np.savez(savepath, train_data, labels)


split_train_test()