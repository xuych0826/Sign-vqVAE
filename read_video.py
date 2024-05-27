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
import datetime


batch_size = 3
txt_file = "/data/rhythmo_36/SignGPT/data/csl/output.txt"
num_cuda = 4
GPU_start = 0
idx_st = 20000
idx_end = 30000

def log_info(txt):
    current_time = datetime.datetime.now().time()
    with open("log.txt", "a") as f:
        f.write(txt + f" time = {current_time}" + "\n")    

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

def calc_index (length, raw_fps, tar_fps):
    factor = raw_fps / tar_fps
    indices = np.arange(0, length, factor).astype(int)
    indices = indices[indices < length]
    return indices
    
def extract_feature(file, device, model):
    log_info(f"{device} is trying to process {file}")

    if not file.endswith(".mp4"):
        log_info(f"{device}:Invalid file: {file}")
        print(f"Invalid file: {file}")
        return
    if not os.path.exists(file):
        log_info(f"{device}:File not found: {file}")
        print(f"File not found: {file}")
        return
    npy_file = file.replace(".mp4", ".npy")
    if os.path.exists(npy_file):
        log_info(f"{device}:File already exists: {npy_file}")
        print(f"File already exists: {npy_file}")
        return
    
    log_info(f"{device} is reading the video {file}")
    root = os.path.dirname(npy_file)
    cap = cv2.VideoCapture(file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log_info(f"{device} extract {count} frames from {file}, fps = {fps}")
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

    log_info(f"{device} is running with {file}")
    whole_features = []
    
    index= calc_index(count-1, fps, 20)
    length = len(index)
    count = 0
    
    for i in trange(0, length, batch_size, desc=f"{device}"):
        imgs = []
        for j in range(batch_size):
            
            ret, img = cap.read()
            if not ret:
                break
            while count not in index:
                count += 1
                ret, img = cap.read()
                if not ret:
                    break
            if not ret:
                break
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transformer(img).unsqueeze(0).to(device)
            imgs.append(img)
            count += 1

        with torch.no_grad():
            if not len(imgs) == 0:
                features = model(torch.cat(imgs, 0))
        whole_features.append(features.cpu().numpy())
        print(features.shape, file, device)
    cap.release()
    whole_features = np.concatenate(whole_features, 0)
    data = {'feature' : whole_features, 'fps' : 20}
    np.save(npy_file, data)
    print(f"Saved features to {npy_file}")
    print("feature :", whole_features.shape, "fps :",fps)
    print("down with", file)
    log_info(f"{device} has down with {file}")

def worker(GPUid, file_list):
    device = torch.device(f"cuda:{GPUid+GPU_start}")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    model.eval()
    model.to(device)
    log_info(f"cuda:{GPUid+GPU_start} has loaded model")
    for file in file_list:
        extract_feature(file, device, model)
    log_info(f"cuda:{GPUid+GPU_start} has down it's work")

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
    random.shuffle(file_list)

    for i in range(num_cuda):
        files = file_list[i::num_cuda]
        p = multiprocessing.Process(target=worker, args=(i,files))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
