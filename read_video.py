import cv2
import torch
import torchvision.transforms as transform
import PIL.Image as Image
import os
import glob
import numpy as np
import random
import multiprocessing

batch_size = 16
txt_file = "/data/rhythmo/Projects/sign_video_new/videos/output.txt"
num_cuda = 8
GPU_start = 0
idx_st = 0
idx_end = 30000


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
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    count = len(frames)
    print(f"Extracted {len(frames)} from {file}")
    H, W, _ = frames[0].shape
    print(f"Image shape: {H}x{W}")
    H = 14 * round(H / 14)
    W = 14 * round(W / 14)
    print(f"Resized image shape: {H}x{W}")
    transformer = transform.Compose([
                transform.Resize((H, W)),
                transform.ToTensor()
            ])
    whole_features = []
    for i in range(0, count, batch_size):
        imgs = []
        for j in range(i, min(count, i + batch_size)):
            img = frames[j]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transformer(img).unsqueeze(0).to(device)
            imgs.append(img)

        with torch.no_grad():
            features = model(torch.cat(imgs, 0))
        whole_features.append(features.cpu().numpy())
        print(features.shape, file, device)
    whole_features = np.concatenate(whole_features, 0)
    np.save(npy_file, whole_features)
    print(f"Saved features to {npy_file}")
    print(whole_features.shape)
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



## 720p, fps, img correct, server