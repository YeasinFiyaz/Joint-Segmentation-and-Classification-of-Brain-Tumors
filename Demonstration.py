# @title
from google.colab import files
uploaded_demo = files.upload()  # choose 2 files: image + mask
demo_files = list(uploaded_demo.keys())
print("Uploaded demo files:", demo_files)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

IMG_SIZE = 256

def preprocess_single_image_gray(img_path, size=IMG_SIZE):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    # normalize to 0-1
    img = img.astype(np.float32) / 255.0
    # to tensor shape (1,1,H,W)
    img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return img, img_t

def overlay_mask(img_gray, mask01, alpha=0.4):
    # img_gray: HxW float [0,1], mask01: HxW {0,1}
    img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1).astype(np.float32)
    color = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # red
    idx = mask01 > 0.5
    img_rgb[idx] = (1-alpha)*img_rgb[idx] + alpha*color
    return img_rgb

def show_demo_grid(original, pred_mask, conf=None):
    """
    Show original, predicted mask, and overlay side-by-side.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    title = ""
    if conf is not None:
        title = f"Class confidence: {conf:.3f}"
    fig.suptitle(title)

    # Original
    axs[0].imshow(original, cmap="gray")
    axs[0].set_title("Original image")
    axs[0].axis("off")

    # Predicted mask
    axs[1].imshow(pred_mask, cmap="gray")
    axs[1].set_title("Predicted mask")
    axs[1].axis("off")

    # Overlay
    axs[2].imshow(overlay_mask(original, pred_mask))
    axs[2].set_title("Overlay")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

@torch.no_grad()
def run_demo_image(model, img_path, device=None):
    """
    Run one image through the model and display prediction.
    model should output (seg_logits, cls_logits).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    orig, img_t = preprocess_single_image_gray(img_path)
    img_t = img_t.to(device)

    seg_logits, cls_logits = model(img_t)

    # predicted mask
    pred_prob = torch.sigmoid(seg_logits).cpu().numpy()[0,0]
    pred_mask = (pred_prob > 0.5).astype(np.float32)

    # classification confidence
    probs = torch.softmax(cls_logits, dim=1).cpu().numpy()[0]
    confidence = float(np.max(probs))

    show_demo_grid(orig, pred_mask, conf=confidence)
    print("Image:", img_path)
    print("Predicted class id:", int(np.argmax(probs)), "| Confidence:", confidence)
MODEL = unet
MODEL = att_unet

demo_images = [
    "1_glioma.jpg",
    "2_meningioma.jpg",
    "3_no.jpg",
    "4_pituitary.jpg"
]

for img in demo_images:
    print("\n=== Demo on:", img, "===")
    run_demo_image(MODEL, img, device=DEVICE)
