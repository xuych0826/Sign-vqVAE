import cv2
import torch
import torchvision.transforms as transform
import PIL.Image as Image
import os
import glob
import numpy as np
import random
from multiprocessing import Process


############################################
batch_size = 16
txt_file = "/data/rhythmo/Projects/sign_video_new/videos/output.txt"
num_cuda = 4
idx_st = 0
idx_end = 1000
############################################

file_list = []
with open(txt_file, "r") as file:
    for line in file:
        file_list.append(line.strip())
print("number of MP4file:",len(file_list))
file_list = file_list[idx_st, idx_end]




