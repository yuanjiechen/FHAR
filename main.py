from pathlib import Path
import torch
import argparse
import logging
import shutil
from datetime import datetime

from util.getlog import get_log
from train import Train
if __name__ == "__main__":

    logger = get_log()
    logger.setLevel(logging.INFO)
    logger.info("New training--------------------\n")
    print(f"[{datetime.now()}] Start new training")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-sl", "--selection", type=str, default="", required=True, help="Select training model and dataset")
    parser.add_argument("-g", "--gpu", type=int, default=0, required=False, help="Select training GPU")
    #parser.add_argument("-t", "--gpu", type=int, default=0, required=False, help="Select training GPU")
    args = parser.parse_args()

    i = 1
    result = Path(f"./result/{args.selection}_{i}.csv")
    while result.exists():
        i += 1
        result = Path(f"./result/{args.selection}_{i}.csv")

    with open(result, "a+"):
        sfx = result.stem
        model_path = result.parent.joinpath(sfx)
        model_path.mkdir()
        shutil.copy("./train_cfg.csv", model_path.joinpath(f"{sfx}_cfg.csv"))

        pass


    print(f"[{datetime.now()}] Result will be saved in {result}")


    t = Train(args)
    t.trainer(result)