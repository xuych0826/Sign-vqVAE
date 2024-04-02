import cv2
import torch
import torchvision.transforms as transform
import PIL.Image as Image
import os
import glob
import numpy as np

batch_size = 2
device = torch.device("cuda:0")
dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
dinov2_vitg14_reg.eval()
dinov2_vitg14_reg.to(device)
print("Model loaded")

video_folder_list = [f"videos/gejianming/生成3",f"videos/gejianming/生成4", f"videos/gejianming/生成5",f"videos/gejianming/CSL7000"]
for video_folder in video_folder_list:
    for root, dirs, files in os.walk(video_folder):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                print(video_path)
                cap = cv2.VideoCapture(video_path)
                success, image = cap.read()
                count = 0
                img_folder = f'frames/{file[:-4]}'
                os.makedirs(img_folder, exist_ok=True)
                while success:
                    cv2.imwrite(f"{img_folder}/frame{count}.jpg", image)
                    success, image = cap.read()
                    count += 1
                cap.release()
            else:
                continue

            print(f"Extracted {count} frames from {file[:-4]}")
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

            
            output_folder = root
            output_path = os.path.join(output_folder, f"{file[:-4]}.npy")
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
                    features = dinov2_vitg14_reg(torch.cat(imgs, 0))
                whole_features.append(features.cpu().numpy())
                print(features.shape, root)
            #save features to file
            whole_features = np.concatenate(whole_features, 0)
            np.save(output_path, whole_features)
            print(f"Saved features to {output_path}")
            print(whole_features.shape)

            #delete the img folder
            for file_name in glob.glob(f'{img_folder}/*'):
                os.remove(file_name)
            print("down with", video_path)
            