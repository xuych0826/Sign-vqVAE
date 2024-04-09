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

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')