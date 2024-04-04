import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
from itertools import cycle

class Dataset(data.Dataset):
    def __init__(self, data_folder, batch_size, window_size):
        self.data_path = data_folder
        self.batch_size = batch_size
        self.window_size = window_size

        self.data = []
        self.lengths = []
        for file in os.listdir(data_folder):
            if file.endswith(".npy"):
                load_data = np.load(pjoin(data_folder, file))
                motion = load_data['feature']
                fps = load_data['fps']
                if motion.shape[0] < self.window_size:
                    continue
                self.data.append(motion)
                self.lengths.append(motion.shape[0]-self.window_size)
        
        print("Total number of motions {}".format(len(self.data)))

    def compute_sampling_prob(self) : 
        
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        motion = self.data[item]
        idx = random.randint(0, len(motion) - self.window_size)
        motion = motion[idx:idx+self.window_size]
        return motion
    
def DATALoader(data_folder, batch_size, window_size, num_workers = 0,):
    trainSet = Dataset(data_folder, batch_size, window_size)
    prob = trainSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(
                                               trainSet,
                                               batch_size=batch_size,
                                               #sampler=sampler,
                                               num_workers=num_workers,
                                               #collate_fn=collate_fn,
                                               drop_last=True)
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


# if __name__ == "__main__":
#     data_folder = "features"
#     batch_size = 1
#     window_size = 64
#     train_loader = DATALoader(data_folder, batch_size, window_size)
#     for i, data in enumerate(train_loader):
#         print(data.shape)
#         if i == 10:
#             break


    
