import cv2
import torch
import torchvision.transforms as transform
import PIL.Image as Image
import os
import glob
import numpy as np
import random

############################################


base_video_folder_list = ["/data/rhythmo/Projects/sign_video_new/videos"]
output_file = "/data/rhythmo/Projects/sign_video_new/videos/output.txt"

file_list = []
for base_video_folder in base_video_folder_list:
    for root, dirs, files in os.walk(base_video_folder):
        for file in files:
            if file.endswith('.mp4'):
                file_list.append(os.join(root, file))

print(len(file_list))

random.shuffle(file_list)

for i in range(10):
    print(file_list[i])

with open(output_file, "w") as file:
    for item in file_list:
        file.write(f"{item}\n")
############################################


file_list = []

with open(output_file, "r") as file:
    for line in file:
        file_list.append(line.strip())

for i in range(10):
    print(file_list[i])