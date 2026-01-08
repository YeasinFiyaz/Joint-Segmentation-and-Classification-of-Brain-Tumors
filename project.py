!pip -q install kaggle tqdm opencv-python

import os, re, glob, random, math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

#File uploading
from google.colab import files
uploaded = files.upload()  # upload brisc2025.zip

import os, zipfile
zip_name = list(uploaded.keys())[0]
print("Uploaded:", zip_name)

os.makedirs("brisc2025", exist_ok=True)
with zipfile.ZipFile(zip_name, 'r') as z:
    z.extractall("brisc2025")

print("Top-level:", os.listdir("brisc2025")[:20])
