import logging

import pynvml
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
RATIO = 1024 * 1024
def gpu_check(gpu_id):
    pynvml.nvmlInit()
    mem_list = []
    gpunum = pynvml.nvmlDeviceGetCount()
    #for gpu_id in range(gpunum):
    if gpu_id >= gpunum or gpu_id < 0:
        return False

    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    mem = (meminfo.free / RATIO)
    if mem > 4500:
        return True
    else :
        return False

