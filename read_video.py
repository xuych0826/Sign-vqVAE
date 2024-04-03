import cv2
import torch
import torchvision.transforms as transform
import PIL.Image as Image
import os
import glob
import numpy as np
import random
from multiprocessing import Process, Manager


############################################
batch_size = 16
txt_file = "/data/rhythmo/Projects/sign_video_new/videos/output.txt"
num_cuda = 1
idx_st = 0
idx_end = 1000
############################################

file_list = ["mp4\Oriental_news_2023-02-13sign.mp4","mp4\Oriental_news_2023-02-14sign.mp4"]
# with open(txt_file, "r") as file:
#     for line in file:
#         file_list.append(line.strip())
# print("number of MP4file:",len(file_list))
# file_list = file_list[idx_st, idx_end]

############################################

def load_model(device):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    model.eval()
    model.to(device)
    return model

models = []
devices = []
for i in range(num_cuda):
    device = torch.device(f"cuda:{i}")
    model = load_model(device)
    devices.append(device)
    models.append(model)


############################################

def extract_feature(file, GPUid):
    if not file.endswith(".mp4"):
        return
    if not os.path.exists(file):
        return
    npy_file = file.replace(".mp4", ".npy")
    if os.path.exists(npy_file):
        return
    root = os.path.dirname(npy_file)
    cap = cv2.VideoCapture(file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    for i in range(0, count//100, batch_size):
        imgs = []
        for j in range(i, min(count//100, i + batch_size)):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transformer(Image.fromarray(frames[j]))
            imgs.append(img).unsqueeze(0).to(devices[GPUid])

            with torch.no_grad():
                features = models[GPUid](torch.cat(imgs, 0))
                whole_features.append(features.cpu().numpy())
                print(features.shape, root)

            whole_features = np.concatenate(whole_features, 0)
            np.save(npy_file, whole_features)
            print(f"Saved features to {npy_file}")
            print(whole_features.shape)
            print("down with", file)


manager = Manager()
unprocessed_elements = manager.list(file_list)

def process_elements(unprocessed_elements, GPUid):
    while unprocessed_elements:
        element = unprocessed_elements.pop(0)
        print(f"Processing element: {element} on GPU {GPUid}")
        extract_feature(element, GPUid)

processes = []
for i in range(num_cuda):
    p = Process(target=process_elements, args=(unprocessed_elements, i))
    p.start()
    processes.append(p)

for p in processes:
    p.join()


    