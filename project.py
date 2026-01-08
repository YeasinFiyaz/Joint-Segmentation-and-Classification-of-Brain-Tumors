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
# @title
import os, glob

# Search common locations for roots
cands = []
for base in ["/content", "."]:
    cands += glob.glob(os.path.join(base, "**/manifest.csv"), recursive=True)
    cands += glob.glob(os.path.join(base, "**/manifest.json"), recursive=True)

print("Found these manifest files:")
for p in cands:
    print(p)

if len(cands) == 0:
    print("\nNo manifest found anywhere. That means either:")
    print("1) You did NOT extract the dataset, or")
    print("2) You extracted a different zip, or")
    print("3) The files are named differently (e.g., manifest without .json/.csv).")
# @title
import os, json
import pandas as pd

ROOT = "/content/brisc2025/brisc2025"
print("ROOT:", ROOT)
print("ROOT files:", os.listdir(ROOT))

manifest_csv  = os.path.join(ROOT, "manifest.csv")
manifest_json = os.path.join(ROOT, "manifest.json")

df = pd.read_csv(manifest_csv)
print("\nLoaded manifest.csv")
print("Columns:", df.columns.tolist())
display(df.head(5))

with open(manifest_json, "r", encoding="utf-8") as f:
    mj = json.load(f)

print("\nLoaded manifest.json")
print("Type:", type(mj))
if isinstance(mj, dict):
    print("Top keys:", list(mj.keys())[:20])
elif isinstance(mj, list):
    print("JSON list length:", len(mj))
    print("First item keys:", list(mj[0].keys())[:20])
