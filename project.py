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
# @title
# class dirs = folders inside classification_task that contain images
class_dirs = []
for d in os.listdir(CLS_ROOT):
    p = os.path.join(CLS_ROOT, d)
    if os.path.isdir(p) and len(collect_images(p)) >= 10:
        class_dirs.append(d)

class_dirs = sorted(class_dirs)
class_to_idx = {c:i for i,c in enumerate(class_dirs)}

print("Detected classes:", class_dirs)
print("class_to_idx:", class_to_idx)

cls_map = {}
for cname, idx in class_to_idx.items():
    files = collect_images(os.path.join(CLS_ROOT, cname))
    for f in files:
        cls_map[stem(f)] = idx

print("Total labeled classification images:", len(cls_map))
# @title
joint_samples = [(im, mk, cls_map[stem(im)]) for im, mk in pairs if stem(im) in cls_map]
print("Joint samples (img+mask+label):", len(joint_samples))

if len(joint_samples) < 50:
    print("\n Joint samples are small.")
    print("That means segmentation file names likely don't match classification file names.")
    print("You can still get full marks by training segmentation and classification separately (allowed).")
# @title
ROOT = "/content/brisc2025/brisc2025"

SEG_IMG_DIR  = f"{ROOT}/segmentation_task/train/images"
SEG_MASK_DIR = f"{ROOT}/segmentation_task/train/masks"

# IMPORTANT FIX: classification root should be train folder
CLS_TRAIN_ROOT = f"{ROOT}/classification_task/train"
CLS_TEST_ROOT  = f"{ROOT}/classification_task/test"  # optional later

print("SEG_IMG_DIR:", SEG_IMG_DIR)
print("SEG_MASK_DIR:", SEG_MASK_DIR)
print("CLS_TRAIN_ROOT:", CLS_TRAIN_ROOT)
print("Exists:", os.path.exists(SEG_IMG_DIR), os.path.exists(SEG_MASK_DIR), os.path.exists(CLS_TRAIN_ROOT))
# @title
IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")

def collect_images(path):
    files=[]
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(path, f"**/*{ext}"), recursive=True)
    return sorted(files)

def stem(p):
    b=os.path.basename(p)
    return os.path.splitext(b)[0]

img_files = collect_images(SEG_IMG_DIR)
mask_files = collect_images(SEG_MASK_DIR)

mask_map = {stem(m): m for m in mask_files}

pairs=[]
missing=0
for im in img_files:
    s=stem(im)
    if s in mask_map:
        pairs.append((im, mask_map[s]))
    else:
        missing += 1

print("Seg train images:", len(img_files))
print("Seg train masks :", len(mask_files))
print("Matched pairs   :", len(pairs))
print("Missing         :", missing)
print("Example pair:", pairs[0])
# @title
def count_imgs(path):
    return sum(len(glob.glob(os.path.join(path, f"**/*{ext}"), recursive=True)) for ext in IMG_EXTS)

# detect class folders under classification_task/train
class_dirs = []
for d in os.listdir(CLS_TRAIN_ROOT):
    p = os.path.join(CLS_TRAIN_ROOT, d)
    if os.path.isdir(p) and count_imgs(p) >= 10:
        class_dirs.append(d)

class_dirs = sorted(class_dirs)
class_to_idx = {c:i for i,c in enumerate(class_dirs)}

print("Detected CLASS folders:", class_dirs)
print("class_to_idx:", class_to_idx)

# map by filename stem -> class id (from folder)
cls_map = {}
for cname, idx in class_to_idx.items():
    files = collect_images(os.path.join(CLS_TRAIN_ROOT, cname))
    for f in files:
        cls_map[stem(f)] = idx

print("Total classification train labeled images:", len(cls_map))
# @title
CODE_TO_CLASSNAME = {
    "gl": "glioma",
    "mn": "meningioma",
    "pi": "pituitary",
    "no": "normal"
}

# Build code->idx mapping based on detected class folder names
# If folder names already match, great; if not, we still map by contains.
code_to_idx = {}
for code, cname in CODE_TO_CLASSNAME.items():
    match = None
    for folder in class_dirs:
        if cname.lower() in folder.lower() or folder.lower() in cname.lower():
            match = folder
            break
    if match is not None:
        code_to_idx[code] = class_to_idx[match]

print("code_to_idx (from folder names):", code_to_idx)

def label_from_filename(path):
    name = os.path.basename(path).lower()
    # find _gl_ or -gl- etc
    m = re.search(r'[_-](gl|mn|pi|no)[_-]', name)
    if m:
        code = m.group(1)
        if code in code_to_idx:
            return code_to_idx[code]
    return None
joint_samples=[]
fallback_used=0

for im, mk in pairs:
    s = stem(im)
    if s in cls_map:
        lab = cls_map[s]
    else:
        lab = label_from_filename(im)
        if lab is None:
            continue
        fallback_used += 1
    joint_samples.append((im, mk, lab))

print("Joint samples:", len(joint_samples))
print("Fallback labels used (from filename):", fallback_used)

IMG_SIZE = 256 #preprocessing    

def read_image_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image: " + path)
    return img

def read_mask_binary(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError("Could not read mask: " + path)
    m = (m > 127).astype(np.uint8)
    return m

def preprocess_img(img, size=IMG_SIZE):  #common preprocessing image pipeline
    img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)/255.0
    return img

def preprocess_mask(mask, size=IMG_SIZE):  #common preprocessing mask pipeline
    mask = cv2.resize(mask, (size,size), interpolation=cv2.INTER_NEAREST)
    mask = (mask>0).astype(np.float32)
    return mask

class SegClsDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples=samples
        self.augment=augment

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        im_path, mk_path, label = self.samples[idx]
        img = preprocess_img(read_image_gray(im_path))
        mk  = preprocess_mask(read_mask_binary(mk_path))

        if self.augment:   #Data Augmentation for best output ensurity
            if random.random() < 0.5:
                img = np.fliplr(img).copy(); mk = np.fliplr(mk).copy()
            if random.random() < 0.5:
                img = np.flipud(img).copy(); mk = np.flipud(mk).copy()

        img_t = torch.from_numpy(img).unsqueeze(0)   # (1,H,W)
        mk_t  = torch.from_numpy(mk).unsqueeze(0)    # (1,H,W)
        y_t   = torch.tensor(label, dtype=torch.long)
        return img_t, mk_t, y_t, im_path, mk_path

def split_samples(samples, val_ratio=0.2):
    idx = np.arange(len(samples))
    np.random.shuffle(idx)
    v = int(len(samples)*val_ratio)
    val_idx = idx[:v]
    tr_idx  = idx[v:]
    tr = [samples[i] for i in tr_idx]
    va = [samples[i] for i in val_idx]
    return tr, va

train_samples, val_samples = split_samples(joint_samples, 0.2)

train_ds = SegClsDataset(train_samples, augment=True)
val_ds   = SegClsDataset(val_samples, augment=False)

BATCH=8
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

NUM_CLASSES = len(class_dirs) if len(class_dirs)>0 else 4
print("Train:", len(train_ds), "Val:", len(val_ds), "NUM_CLASSES:", NUM_CLASSES)
