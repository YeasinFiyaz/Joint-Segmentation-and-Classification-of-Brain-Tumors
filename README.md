## Joint-Segmentation-and-Classification-of-Brain-Tumors
This project implements a deep learning–based system for brain tumor segmentation and classification from MRI images using U-Net and Attention U-Net architectures. The work was carried out as part of the CSE428: Image Processing course project.

**Project Objectives**

- Perform pixel-level brain tumor segmentation using U-Net
- Attach a classification head to the encoder to predict tumor type
- Upgrade the model to Attention U-Net and analyze performance improvement
- Train and evaluate models using proper training, validation, and test sets
- Visualize predictions for any single input image

**Dataset**
**Dataset: BRISC2025 Brain MRI Dataset**


**Tasks:**
- Segmentation (MRI image + tumor mask)
- Classification (tumor type)

**Tumor classes:**

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

**Preprocessing Pipeline**

A common preprocessing pipeline was applied to all images:
- Resize images and masks to 256 × 256
- Normalize pixel values to [0, 1]
- Convert masks to binary format
- Apply data augmentation (random horizontal and vertical flips) during training

**Model Architectures**
🔹 U-Net with Classifier Head
- ***Encoder–decoder architecture for segmentation***
- ***Skip connections to preserve spatial details***
- ***MLP classifier head attached to the encoder bottleneck for tumor classification***

🔹 Attention U-Net with Classifier Head
- ***Extends U-Net by adding attention gates in skip connections***
- ***Focuses on tumor-relevant regions and suppresses background noise***
- ***Improves segmentation accuracy for small tumor regions***

**Tasks Performed**
- Segmentation
- Binary tumor vs background segmentation

**Metrics used:**

- Mean Intersection over Union (mIoU)
- Dice coefficient
- Pixel accuracy
- Classification
- Image-level tumor type prediction

**Metrics used:**

- Accuracy
- Precision
- Recall
- F1-score

**Training Strategy**

Joint training of segmentation and classification

Combined loss:

Segmentation loss + weighted classification loss

Optimizer: Adam

Learning rate: 1e-3

Epochs: 10–30 (configurable)

##Visualization & Demo

The project includes a single-image inference pipeline that:

Takes any MRI image as input

Produces:

Predicted tumor mask

Overlay visualization

Predicted tumor class with confidence

# Results Summary

Attention U-Net showed better segmentation performance than standard U-Net

Attention mechanism improved focus on tumor regions

Joint learning enabled efficient feature sharing between tasks

**Tools & Technologies**

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Google Colab

# This project is intended for academic and educational purposes.
