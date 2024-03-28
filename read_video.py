import cv2
import torch
import torchvision.transforms as transform
import PIL.Image as Image
import os
import glob
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vitl14.eval()
dinov2_vitl14.to(device)

video_folder = 'mp4'
for file_name in os.listdir(video_folder):
    if file_name.endswith('.mp4'):
        video_path = os.path.join(video_folder, file_name)
        cap = cv2.VideoCapture(video_path)
        success, image = cap.read()
        count = 0
        img_folder = f'frames/{file_name[:-4]}'
        os.makedirs(img_folder, exist_ok=True)
        while success:
            cv2.imwrite(f"{img_folder}/frame{count}.jpg", image)
            success, image = cap.read()
            count += 1
        cap.release()
    else:
        continue

    print(f"Extracted {count} frames from {file_name[:-4]}")
    #get the H,W of the first frame
    img = cv2.imread(f'{img_folder}/frame0.jpg')
    print(f"Image shape: {img.shape}")
    H, W, _ = img.shape
    print(f"Image shape: {H}x{W}")
    #let's H,W be divisible by 14(rounded to the nearest multiple of 14)
    H = 14 * round(H / 14)
    W = 14 * round(W / 14)
    print(f"Resized image shape: {H}x{W}")
    transformer = transform.Compose([
        transform.Resize((H, W)),
        transform.ToTensor()
    ])

    batch_size = 128
    output_folder = 'features'
    output_path = os.path.join(output_folder, f"{file_name[:-4]}.npy")
    whole_features = []

    for i in range(0, count, batch_size):
        imgs = []
        for j in range(i, min(count, i + batch_size)):
            img = cv2.imread(f'{img_folder}/frame{j}.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = transformer(img).unsqueeze(0).to(device)
            imgs.append(img)

        with torch.no_grad():
            features = dinov2_vitl14(torch.cat(imgs, 0))
        whole_features.append(features.cpu().numpy())
        print(features.shape)
    #save features to file
    whole_features = np.concatenate(whole_features, 0)
    np.save(output_path, whole_features)
    print(f"Saved features to {output_path}")
    print(whole_features.shape)

    #delete the img folder
    for file_name in glob.glob(f'{img_folder}/*'):
        os.remove(file_name)
    print("down with", video_path)
        


# video_path = 'mp4/sign.mp4'
# cap = cv2.VideoCapture(video_path)
# success, image = cap.read()
# count = 0

# while success:
#     cv2.imwrite("frames/frame%d.jpg" % count, image)    
#     success, image = cap.read()
#     count += 1

# cap.release()

# transform = transform.Compose([
#     transform.Resize((210, 210)),
#     transform.ToTensor()
# ])

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# pretrained_weights = torch.load('dinov2_vitl14_pretrain.pth')
# dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# dinov2_vitl14.eval()
# dinov2_vitl14.to(device)

# print("count=", count)

# batch_size = 128

# for i in range(0, count, batch_size):
#     imgs = []
#     for j in range(i, min(count, i + batch_size)):
#         img = cv2.imread(f'frames/frame{j}.jpg')
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(img)
#         img = transform(img).unsqueeze(0).to(device)
#         imgs.append(img)

#     with torch.no_grad():
#         features = dinov2_vitl14(torch.cat(imgs, 0))
#     #save features to file
#     torch.save(features, f'features/features{i}.pth')

#     print(features.shape)
