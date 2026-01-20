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
def iou_score_from_logits(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    targets = (targets > 0.5).float()
    inter = (preds*targets).sum(dim=(1,2,3))
    union = (preds + targets - preds*targets).sum(dim=(1,2,3))
    return ((inter+eps)/(union+eps)).mean().item()

def cls_accuracy(logits, labels):
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean().item()

def overlay_mask(img_gray, mask01, alpha=0.4):
    img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32)
    color = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    idx = mask01 > 0.5
    img_rgb[idx] = (1-alpha)*img_rgb[idx] + alpha*color
    return img_rgb

def show_result_grid(original, gt_mask, pred_mask, processed=None, cls_acc=None, iou=None):
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    title = "classification accuracy - "
    title += f"{cls_acc:.3f} " if cls_acc is not None else "- "
    title += "   IoU - "
    title += f"{iou:.3f}" if iou is not None else "-"
    fig.suptitle(title)

    axs[0,0].imshow(original, cmap="gray"); axs[0,0].set_title("Original image"); axs[0,0].axis("off")
    axs[0,1].imshow(gt_mask, cmap="gray"); axs[0,1].set_title("Original mask"); axs[0,1].axis("off")
    axs[0,2].imshow(overlay_mask(original, gt_mask)); axs[0,2].set_title("Original image with mask overlay"); axs[0,2].axis("off")

    if processed is None:
        processed = original
    axs[1,0].imshow(processed, cmap="gray")
    axs[1,0].set_title("Processed image (if applicable else just show original image)")
    axs[1,0].axis("off")

    axs[1,1].imshow(pred_mask, cmap="gray"); axs[1,1].set_title("Predicted mask"); axs[1,1].axis("off")
    axs[1,2].imshow(overlay_mask(original, pred_mask)); axs[1,2].set_title("Original image with predicted mask overlay"); axs[1,2].axis("off")

    plt.tight_layout()
    plt.show()
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self,x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self,x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, 2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetWithClassifier(nn.Module):
    def __init__(self, in_ch=1, num_classes=4, base=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base) #Downsampling
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.down4 = Down(base*8, base*16)

        self.up1 = Up(base*16, base*8)
        self.up2 = Up(base*8,  base*4)
        self.up3 = Up(base*4,  base*2)
        self.up4 = Up(base*2,  base)
        self.outc = nn.Conv2d(base, 1, 1)

        # classifier head from bottleneck (encoder output)  #custom 'MLP classifier head' (Random classifier as asked in the project guideline)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base*16, base*8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(base*8, num_classes)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.down4(x4)

        cls_logits = self.cls_head(self.gap(xb))

        x = self.up1(xb, x4)  #Decoder: up sampling
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        seg_logits = self.outc(x)
        return seg_logits, cls_logits
def train_joint(model, train_loader, val_loader, epochs=10, lr=1e-3, lambda_cls=0.5):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    seg_loss_fn = nn.BCEWithLogitsLoss()
    cls_loss_fn = nn.CrossEntropyLoss()

    hist = {k: [] for k in [
        "train_total_loss","val_total_loss",
        "train_seg_loss","val_seg_loss",
        "train_cls_loss","val_cls_loss",
        "train_iou","val_iou",
        "train_acc","val_acc"
    ]}

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type=="cuda"))

    for ep in range(epochs):
        model.train()
        sums = {"tot":0,"seg":0,"cls":0,"iou":0,"acc":0,"n":0}

        for imgs, masks, labels, _, _ in tqdm(train_loader, desc=f"Train {ep+1}/{epochs}"):
            imgs, masks, labels = imgs.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type=="cuda")):
                seg_logits, cls_logits = model(imgs)
                seg_loss = seg_loss_fn(seg_logits, masks)
                cls_loss = cls_loss_fn(cls_logits, labels)
                loss = seg_loss + lambda_cls*cls_loss

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = imgs.size(0)
            sums["tot"] += loss.item()*bs
            sums["seg"] += seg_loss.item()*bs
            sums["cls"] += cls_loss.item()*bs
            sums["iou"] += iou_score_from_logits(seg_logits.detach(), masks.detach())*bs
            sums["acc"] += cls_accuracy(cls_logits.detach(), labels.detach())*bs
            sums["n"] += bs

        hist["train_total_loss"].append(sums["tot"]/sums["n"])
        hist["train_seg_loss"].append(sums["seg"]/sums["n"])
        hist["train_cls_loss"].append(sums["cls"]/sums["n"])
        hist["train_iou"].append(sums["iou"]/sums["n"])
        hist["train_acc"].append(sums["acc"]/sums["n"])

        model.eval()
        sums = {"tot":0,"seg":0,"cls":0,"iou":0,"acc":0,"n":0}
        with torch.no_grad():
            for imgs, masks, labels, _, _ in tqdm(val_loader, desc=f"Val {ep+1}/{epochs}"):
                imgs, masks, labels = imgs.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
                seg_logits, cls_logits = model(imgs)
                seg_loss = seg_loss_fn(seg_logits, masks)
                cls_loss = cls_loss_fn(cls_logits, labels)
                loss = seg_loss + lambda_cls*cls_loss

                bs = imgs.size(0)
                sums["tot"] += loss.item()*bs
                sums["seg"] += seg_loss.item()*bs
                sums["cls"] += cls_loss.item()*bs
                sums["iou"] += iou_score_from_logits(seg_logits, masks)*bs
                sums["acc"] += cls_accuracy(cls_logits, labels)*bs
                sums["n"] += bs

        hist["val_total_loss"].append(sums["tot"]/sums["n"])
        hist["val_seg_loss"].append(sums["seg"]/sums["n"])
        hist["val_cls_loss"].append(sums["cls"]/sums["n"])
        hist["val_iou"].append(sums["iou"]/sums["n"])
        hist["val_acc"].append(sums["acc"]/sums["n"])

        print(f"\nEpoch {ep+1}/{epochs} | "
              f"Train loss {hist['train_total_loss'][-1]:.4f} IoU {hist['train_iou'][-1]:.4f} Acc {hist['train_acc'][-1]:.4f} | "
              f"Val loss {hist['val_total_loss'][-1]:.4f} IoU {hist['val_iou'][-1]:.4f} Acc {hist['val_acc'][-1]:.4f}\n")

    return hist

def plot_hist(hist, prefix=""):
    for k,v in hist.items():
        plt.figure(figsize=(6,4))
        plt.plot(v)
        plt.title(prefix + k)
        plt.xlabel("epoch")
        plt.grid(True)
        plt.show()
#U NET
unet = UNetWithClassifier(in_ch=1, num_classes=NUM_CLASSES, base=32)
hist_unet = train_joint(unet, train_loader, val_loader, epochs=10, lr=1e-3, lambda_cls=0.5)

plot_hist(hist_unet, prefix="U-Net ")

@torch.no_grad()
def show_random_result(model, dataset):
    model.eval()
    idx = random.randint(0, len(dataset)-1)
    img_t, mk_t, y_t, im_path, mk_path = dataset[idx]

    seg_logits, cls_logits = model(img_t.unsqueeze(0).to(DEVICE))
    pred = (torch.sigmoid(seg_logits).cpu().numpy()[0,0] > 0.5).astype(np.float32)

    img = img_t.squeeze(0).numpy()
    gt  = mk_t.squeeze(0).numpy()

    iou = iou_score_from_logits(seg_logits.cpu(), mk_t.unsqueeze(0))
    acc = cls_accuracy(cls_logits.cpu(), y_t.unsqueeze(0))

    show_result_grid(img, gt, pred, processed=None, cls_acc=acc, iou=iou)

    print("Image:", im_path)
    if len(class_dirs)>0:
        print("Pred class id:", cls_logits.argmax(dim=1).item(), "| True id:", y_t.item())

show_random_result(unet, val_ds)
show_random_result(unet, val_ds)

for _ in range(10):
    show_random_result(unet, val_ds)
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi

class UpAtt(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, 2)
        self.att = AttentionGate(F_g=in_ch//2, F_l=skip_ch, F_int=out_ch)
        self.conv = DoubleConv(in_ch//2 + skip_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x2 = self.att(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionUNetWithClassifier(nn.Module):
    def __init__(self, in_ch=1, num_classes=4, base=32):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.down4 = Down(base*8, base*16)

        self.up1 = UpAtt(base*16, base*8, base*8)
        self.up2 = UpAtt(base*8,  base*4, base*4)
        self.up3 = UpAtt(base*4,  base*2, base*2)
        self.up4 = UpAtt(base*2,  base,   base)
        self.outc = nn.Conv2d(base, 1, 1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base*16, base*8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(base*8, num_classes)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.down4(x4)

        cls_logits = self.cls_head(self.gap(xb))

        x = self.up1(xb, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        seg_logits = self.outc(x)
        return seg_logits, cls_logits

att_unet = AttentionUNetWithClassifier(in_ch=1, num_classes=NUM_CLASSES, base=32)
hist_att = train_joint(att_unet, train_loader, val_loader, epochs=10, lr=1e-3, lambda_cls=0.5)

plot_hist(hist_att, prefix="Attention U-Net ")

print("U-Net best val IoU:", max(hist_unet["val_iou"]))
print("Att best val IoU:", max(hist_att["val_iou"]))
print("U-Net best val Acc:", max(hist_unet["val_acc"]))
print("Att best val Acc:", max(hist_att["val_acc"]))

show_random_result(att_unet, val_ds)
show_random_result(att_unet, val_ds)

for _ in range(10):
    show_random_result(att_unet, val_ds)
