from PIL import Image
from pathlib import Path
import numpy as np
import re

img_path = Path("/mnt/ssd1/fooddataset/box_heatmaps/")
save_path = Path("/home/yuanjie/heatmaps/")
pt = re.compile(r'(?<=a)[0-9]{2}|(?<=s)[0-9]{2}')

def cuts():
    for f in img_path.rglob("*.png"):
        img = Image.open(f)
        img = img.crop((144, 50, 1037, 236))
        img.save(f)
        

def split_train_test():
    hz = 30
    timestep = 0.5
    window = 5

    dic = {0:0, 3:1, 6:2, 7:3, 8:4, 9:5, 10:6, 11:7, 12:8}
    data_count = 0
    for fd in img_path.glob("*"):
        if fd.name in ['train', "test"]:
            continue
        else:
            act, subj = pt.findall(fd.name)
            act = dic[int(act) - 1]
            subj = int(subj)
            print(fd.name)

            img_list = sorted(fd.glob("*.npz"), key=lambda x:int(x.stem))

            i = 0
            one_data = []
            while (i + hz * window) < len(img_list):
                sample = []
                tmp_list = img_list[i:i + hz * window]  
                for image in tmp_list:
                    img = np.load(image)["arr_0"]
                    #print(img.shape)
                    #img = np.fromfile(image).reshape((32, 128))
                    #print(img.shape)
                    #return
                    sample.append(img)
                
                one_data.append(np.stack(sample, axis=0))
                i += int(timestep * hz)
            
            train_data = np.stack(one_data, axis=0)
            labels = np.repeat(act, train_data.shape[0])

            if subj < 10:
                savepath = save_path.joinpath(f"train/{data_count}.npz")

            else:
                savepath = save_path.joinpath(f"test/{data_count}.npz")
            
            data_count += 1
            print(train_data.shape)
            np.savez(savepath, train_data, labels)


split_train_test()