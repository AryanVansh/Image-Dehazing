# 🌫️ Image Dehazing using GAN and Traditional Methods

This project implements single image and video dehazing using two complementary approaches:

 1. Deep Learning-based GAN model trained to translate hazy images into dehazed ones.

 2. Traditional Dark Channel Prior (DCP) method for image restoration.

 3. Guided Filter (GF) implementation for edge-preserving smoothing and refining transmission maps.

The code is modular, covering:

 1.Training and applying a GAN for image dehazing.

 2.DCP-based single image dehazing.

 3.Guided filtering to improve results.

## 📁 Directory Structure
```
.
├── ganmodel.py         # PyTorch GAN for image dehazing (training + video inference)
├── haze_removal.py     # Traditional dehazing using Dark Channel Prior
├── gf.py               # Guided filter for smoothing/filtering
├── generator.pth       # (Generated after training) Trained generator weights
└── README.md           # Project documentation
```

## 🔧 Prerequisites
### Python ≥ 3.7

- PyTorch

- torchvision

- OpenCV

- NumPy

- SciPy

- Matplotlib

- Install dependencies using pip:
```
pip install torch torchvision opencv-python numpy scipy matplotlib
```

## ✅ requirements

 1. requirements.txt

This file lists all dependencies needed to run the project:

```
torch
torchvision
opencv-python
numpy
scipy
matplotlib
```
Save the above content in a file named requirements.txt.

Then users can install dependencies using:
```
pip install -r requirements.txt
```
2. train_test_split/ Dataset Structure Guide
Since your GAN training relies on paired images (hazy + clear), here’s a recommended folder structure:
```
dataset/
├── hazy/
│   ├── 1_hazy.jpg
│   ├── 2_hazy.jpg
│   └── ...
├── clear/
│   ├── 1_clear.jpg
│   ├── 2_clear.jpg
│   └── ...
```
If you're using torchvision.datasets.ImageFolder, you may want to merge both sets into a shared folder like this:
```
train_data/
├── hazy/
│   └── [all hazy images]
├── clear/
│   └── [all clear images]
```
Then in ganmodel.py, use:
```
train_dataset = datasets.ImageFolder(
    root='train_data',
    transform=transform
)
```
## 🎯 Objectives
  - Restore visibility in hazy images.

  - Train a GAN on hazy/clear image pairs.

  - Optionally use traditional methods for comparison or enhancement.

## 🧠 GAN-based Dehazing (ganmodel.py)
### Architecture
Generator: A symmetric encoder-decoder architecture with convolutional and transposed convolutional layers.

Discriminator: A binary classifier with convolutional layers to distinguish real vs. generated images.

###  Features
- Uses BCELoss for adversarial training and L1Loss for content preservation.

- Image preprocessing includes resizing, normalization, and batching.

- Trains on paired datasets using PyTorch's ImageFolder.

###  Training
Dataset Structure

Place hazy and clear images in two subfolders under a root directory like this:
```
path_to_dataset/
├── hazy/
│   ├── 1_hazy.jpg
│   ├── ...
├── clear/
│   ├── 1_clear.jpg
│   ├── ...
```
Edit ganmodel.py to set the correct path:
```
train_dataset = datasets.ImageFolder(root='path_to_dataset', transform=transform)
```
Run:
```
python ganmodel.py
```
During training, logs such as generator and discriminator loss will be printed:
```
[Epoch 0/100] [Batch 0/XYZ] [D loss: 0.632] [G loss: 0.948]
```
After training, the model is saved as generator.pth.

## 🎥 Video Dehazing
The trained generator can be used to dehaze videos.

Update in ganmodel.py:
```
input_video_path = 'hazy.mp4'
output_video_path = 'dehazed.mp4'
generator.load_state_dict(torch.load('generator.pth'))
```

Run:
```
python ganmodel.py
```
This will load the input video, apply the trained generator frame-by-frame, and save the output video.


## 🧪 Traditional Dehazing - Dark Channel Prior (haze_removal.py)

### 📚 Pipeline

- Dark Channel Estimation
Lowest pixel intensity across color channels + erosion.

- Atmospheric Light Estimation
Brightest pixels from dark channel.

- Transmission Map
Estimated using normalized image and omega parameter.

- Scene Radiance Recovery
Final dehazed image using estimated transmission and atmospheric light.

## 📸 Example Usage
Update image path in the file:
```
image_path = "D:/Projects/Dehaze/hazy/29_hazy.png"
```

Run:
```
python haze_removal.py
```
The script will visualize:

- Original image

- Dark channel

- Transmission map

- Final dehazed image

## 🧼 Guided Filter (gf.py)
### 📚 Purpose
The Guided Filter smooths and refines images while preserving edges, useful especially for:

- Transmission map refinement.

- Post-processing results of the DCP.

### 🧪 Run Test
Test filters on example images by running:
```
python -c "import gf; gf.test_gf()"
```
This will:

- Smooth images using guided filtering.

- Save results to disk like cat_smoothed.png, tulips_smoothed.png, etc.

### 📊 Results
You can compare GAN and traditional method outputs qualitatively by visualizing them. Add dehazed samples to your GitHub repo for better demonstration.

### 📝 Notes & Suggestions
- The current GAN training pipeline assumes paired datasets (hazy → clear).

- For video dehazing, make sure OpenCV can read and write to your specified format.

- Enhance results by combining DCP and GAN approaches or post-processing transmission maps with gf.py.
