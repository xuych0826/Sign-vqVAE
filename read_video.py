import cv2
import torch
import torchvision.transforms as transform
import PIL.Image as Image
import os
import glob
import numpy as np
import random
import multiprocessing
from tqdm import trange


batch_size = 10
txt_file = "/data/rhythmo/SignGPT/data/online/output.txt"
num_cuda = 8
GPU_start = 0
idx_st = 0
idx_end = 2000

def calc_size(original_height, original_width, max_size = 720):
    new_height, new_width = 0, 0
    if original_width <= max_size and original_height <= max_size:
        new_height, new_width = original_height, original_width

    aspect_ratio = original_width / original_height
    if original_width >= original_height:
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(max_size * aspect_ratio)

    new_height = 14 * round(new_height / 14)
    new_width = 14 * round(new_width / 14)

    return new_height, new_width

def extract_feature(file, device, model):

    if not file.endswith(".mp4"):
        print(f"Invalid file: {file}")
        return
    if not os.path.exists(file):
        print(f"File not found: {file}")
        return
    npy_file = file.replace(".mp4", ".npy")
    if os.path.exists(npy_file):
        print(f"File already exists: {npy_file}")
        return
    
    root = os.path.dirname(npy_file)
    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"extract {count} frames from {file}")

    ret, frame = cap.read()
    H, W, _ = frame.shape
    print(f"Image shape: {H}x{W}")
    H, W = calc_size(H,W)
    print(f"Resized image shape: {H}x{W}")
    transformer = transform.Compose([
                transform.Resize((H, W)),
                transform.ToTensor()
            ])

    whole_features = []
    for i in trange(0, count-1, batch_size, desc=f"{device}"):
        imgs = []
        for j in range(batch_size):
            ret, img = cap.read()
            if not ret:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transformer(img).unsqueeze(0).to(device)
            imgs.append(img)

        with torch.no_grad():
            features = model(torch.cat(imgs, 0))
        whole_features.append(features.cpu().numpy())
        print(features.shape, file, device)
    cap.release()
    whole_features = np.concatenate(whole_features, 0)
    data = {'feature' : whole_features, 'fps' : fps}
    np.save(npy_file, data)
    print(f"Saved features to {npy_file}")
    print("feature :", whole_features.shape, "fps :",fps)
    print("down with", file)

def worker(GPUid, file_list):
    device = torch.device(f"cuda:{GPUid+GPU_start}")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    model.eval()
    model.to(device)
    for file in file_list:
        extract_feature(file, device, model)



if __name__ == '__main__':

    file_list = []
    with open(txt_file, "r") as file:
        for line in file:
            file_list.append(line.strip())
    print("number of MP4file:",len(file_list))
    file_list = file_list[idx_st:idx_end]

    print("len=:",len(file_list))
    count = len(file_list)
    processes = []

    for i in range(num_cuda):
        files = file_list[i::num_cuda]
        p = multiprocessing.Process(target=worker, args=(i,files))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()



##fps, server