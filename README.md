# STL-10 Classification with Neural Networks

## Overview
This project implements multi-class classification on the STL-10 dataset using PyTorch with five different neural network architectures:
1. **Logistic Regression** (baseline)
2. **Fully Connected Neural Network** with 3 hidden layers
3. **Convolutional Neural Network (CNN)** 
4. **Fixed Pre-trained MobileNetV2** (feature extractor only)
5. **Fine-tuned Pre-trained MobileNetV2** (end-to-end training)

The STL-10 dataset consists of 96x96 pixel RGB images across 10 classes: Airplane, Bird, Car, Cat, Deer, Dog, Horse, Monkey, Ship, and Truck, with 500 training images and 800 test images per class.

## Requirements
```
torch
torchvision
numpy
matplotlib
sklearn
```

## How to Run

### Prerequisites
1. **Python Version:** 3.8.18
2. **Environment:** Jupyter Notebook required (.ipynb format)
3. **Hardware:** GPU recommended (Google Colab or similar)
4. Install required packages:
   ```bash
   pip install torch torchvision numpy matplotlib scikit-learn
   ```
   
## Data Preprocessing

### Image Transformations
**Training Set:**
- Random cropping to 64x64 pixels
- Random horizontal flipping (data augmentation)
- Random rotation (±20 degrees)
- Random affine transformation (±5 degrees)

**Test Set:**
- Center cropping to 64x64 pixels
- Standard normalization

### Data Splits
- **Training:** 80% of original training data
- **Validation:** 20% of original training data  
- **Test:** Original test set (800 images per class)

## Model Architectures

### 1. Logistic Regression
- **Input:** Flattened 64×64×3 images (12,288 features)
- **Architecture:** Single linear layer with softmax
- **Hyperparameters:** LR=5e-05, SGD optimizer, L2=0.01, 150 epochs

### 2. Fully Connected Neural Network
- **Architecture:** 3 hidden layers [200, 100, 50] + classification layer
- **Features:** Batch normalization and dropout on all hidden layers
- **Total Parameters:** 2,484,160
- **Hyperparameters:** LR=0.001, SGD optimizer, L2=0.01, 200 epochs

### 3. Convolutional Neural Network
- **Architecture:** 2 Conv layers + 2 Pooling + 2 FC layers
- **Conv Layers:** [20, 64] channels with 3×3 kernels
- **FC Layers:** [100, 50] neurons
- **Features:** Batch normalization on conv layers, dropout on FC layers
- **Total Parameters:** 1,272,372
- **Hyperparameters:** LR=0.0001, Adam optimizer, L2=0.0001, 200 epochs

### 4. Fixed Pre-trained MobileNetV2
- **Feature Extractor:** Pre-trained MobileNetV2 (frozen weights)
- **Task Head:** 2 FC layers [custom, 10] + classification
- **Training:** Only task head parameters updated
- **Hyperparameters:** LR=0.00015, Adam optimizer, L2=5e-05, 100 epochs

### 5. Fine-tuned Pre-trained MobileNetV2  
- **Feature Extractor:** Pre-trained MobileNetV2 (trainable weights)
- **Task Head:** 2 FC layers [custom, 10] + classification
- **Training:** End-to-end fine-tuning of entire network
- **Hyperparameters:** LR=0.0001, SGD optimizer, L2=4e-05, 100 epochs

## Results Summary

| Model | Train Acc | Val Acc | Test Acc | Train Loss | Val Loss | Test Loss |
|-------|-----------|---------|----------|------------|----------|-----------|
| Logistic Regression | 26.0% | 27.8% | 28.1% | 2.126 | 2.021 | 2.015 |
| Fully Connected NN | 29.4% | 36.2% | 35.4% | 1.981 | 1.798 | 1.805 |
| CNN | 54.3% | 60.0% | 59.7% | 1.288 | 1.093 | 1.129 |
| Fixed MobileNetV2 | 64.3% | 66.1% | 64.3% | 1.035 | 0.949 | 1.013 |
| **Fine-tuned MobileNetV2** | **79.9%** | **82.8%** | **80.9%** | **0.571** | **0.505** | **0.560** |

## Key Findings

### Performance Insights
1. **Transfer Learning Superiority:** Pre-trained models significantly outperformed models trained from scratch
2. **CNN vs FC:** CNN achieved much better performance than fully connected networks for image classification
3. **Fine-tuning Benefits:** End-to-end fine-tuning of MobileNetV2 provided the best results
4. **Data Augmentation Impact:** Random transformations improved generalization across all models

### Hyperparameter Insights
- **Batch Size:** 128 provided optimal balance between training stability and speed
- **Learning Rate:** Lower rates (1e-4 to 1e-5) worked better for pre-trained models
- **Regularization:** L2 values between 4e-05 and 0.01 prevented overfitting effectively
- **Optimizers:** Adam performed better for CNN and transfer learning models; SGD for simpler architectures
