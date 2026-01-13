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
# @title
import os, glob

IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")

def collect_images(path):
    files=[]
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(path, f"**/*{ext}"), recursive=True)
    return sorted(files)

SEG_ROOT = os.path.join(ROOT, "segmentation_task")
CLS_ROOT = os.path.join(ROOT, "classification_task")

print("SEG_ROOT exists:", os.path.exists(SEG_ROOT), SEG_ROOT)
print("CLS_ROOT exists:", os.path.exists(CLS_ROOT), CLS_ROOT)

print("Seg images found:", len(collect_images(SEG_ROOT)))
print("Cls images found:", len(collect_images(CLS_ROOT)))
# @title
import re

def stem(p):
    b = os.path.basename(p)
    return os.path.splitext(b)[0]

def count_imgs(p):
    return sum(len(glob.glob(os.path.join(p, f"**/*{ext}"), recursive=True)) for ext in IMG_EXTS)

# find candidate dirs inside segmentation_task
dirs=[]
for p, d, f in os.walk(SEG_ROOT):
    c = count_imgs(p)
    if c >= 10:
        dirs.append((p,c))
dirs = sorted(dirs, key=lambda x: x[1], reverse=True)

mask_like = [(p,c) for p,c in dirs if any(k in p.lower() for k in ["mask","gt","label","seg"])]
img_like  = [(p,c) for p,c in dirs if any(k in p.lower() for k in ["image","img","scan","data"])]

print("Top image-like dirs:")
for p,c in img_like[:5]:
    print(c, ":", p)

print("\nTop mask-like dirs:")
for p,c in mask_like[:5]:
    print(c, ":", p)

SEG_IMG_DIR  = img_like[0][0] if len(img_like)>0 else SEG_ROOT
SEG_MASK_DIR = mask_like[0][0] if len(mask_like)>0 else SEG_ROOT

print("\nChosen SEG_IMG_DIR :", SEG_IMG_DIR)
print("Chosen SEG_MASK_DIR:", SEG_MASK_DIR)

img_files  = collect_images(SEG_IMG_DIR)
mask_files = collect_images(SEG_MASK_DIR)

mask_map = {stem(m): m for m in mask_files}

pairs=[]
missing=0
for im in img_files:
    s = stem(im)
    if s in mask_map:
        pairs.append((im, mask_map[s]))
    else:
        s2 = re.sub(r'(_mask|_seg|_label|_gt)$', '', s, flags=re.IGNORECASE)
        if s2 in mask_map:
            pairs.append((im, mask_map[s2]))
        else:
            missing += 1

print("\nTotal images:", len(img_files))
print("Total masks :", len(mask_files))
print("Matched pairs:", len(pairs))
print("Missing matches:", missing)

print("\nSample pair:", pairs[0] if len(pairs)>0 else None)
