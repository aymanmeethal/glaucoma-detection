# Glaucoma Detection Using Deep Learning

## Overview

Glaucoma is one of the leading causes of irreversible blindness worldwide. Early diagnosis is critical for preventing vision loss and improving patient outcomes.

This project presents an automated glaucoma detection system that analyzes retinal fundus images using deep learning and transfer learning techniques. The system classifies retinal images as either glaucoma-positive or glaucoma-negative and provides confidence scores for predictions.

The model utilizes EfficientNetV2L pretrained on ImageNet along with retinal image enhancement and data augmentation techniques to improve classification performance.

Dataset sourced from KJ Somaiya Hospital under institutional permission for academic and research purposes. The dataset is not included in this repository due to privacy and data-sharing restrictions.

---

## Features

* Automated glaucoma screening from retinal fundus images
* Transfer learning using EfficientNetV2L
* CLAHE-based image enhancement
* Data augmentation for improved generalization
* Binary classification:

  * Glaucoma Positive
  * Glaucoma Negative
* Confidence-based predictions
* Single-image and batch-image inference
* Model saving and reloading support

---

## Dataset

The retinal fundus image dataset used in this project was obtained from **KJ Somaiya Hospital** under institutional permission for academic and research purposes.

Due to privacy, ethical, and data-sharing restrictions, the dataset is not included in this repository.

The dataset consists of retinal fundus images categorized into:

* Glaucoma Positive
* Glaucoma Negative

---

## Methodology

### 1. Image Preprocessing

Retinal images can vary significantly in illumination and contrast. To improve image quality, the project applies:

#### CLAHE (Contrast Limited Adaptive Histogram Equalization)

* Converts image to HSV color space
* Applies CLAHE on the V (brightness) channel
* Enhances local contrast while reducing noise amplification
* Improves visibility of retinal structures

---

### 2. Data Augmentation

To improve model robustness and reduce overfitting:

* Random Rotation (±30°)
* Horizontal Flipping
* Vertical Flipping

---

### 3. Deep Learning Model

#### Base Network

EfficientNetV2L (ImageNet Pretrained)

Transfer learning is used to leverage features learned from large-scale image datasets.

#### Classification Head

* Global Average Pooling Layer
* Flatten Layer
* Dense Layer (512 neurons)
* Dense Layer (256 neurons)
* Dense Layer (128 neurons)
* Softmax Output Layer (2 classes)

---

## Training Configuration

| Parameter     | Value                    |
| ------------- | ------------------------ |
| Input Size    | 300 × 300 × 3            |
| Batch Size    | 32                       |
| Optimizer     | Adam                     |
| Learning Rate | 0.001                    |
| Loss Function | Categorical Crossentropy |
| Classes       | 2                        |
| Framework     | TensorFlow / Keras       |

---

## Project Workflow

1. Load training and validation retinal image datasets
2. Perform optional CLAHE image enhancement
3. Apply data augmentation
4. Extract deep features using EfficientNetV2L
5. Train custom classification layers
6. Save trained model as `GlaucomaDetection.h5`
7. Predict glaucoma status on unseen retinal images
8. Display prediction confidence

---

## Folder Structure

glaucoma-detection/

├── main.py

├── train_model.py

├── prediction.py

├── image_preprocess.py

├── plot.py

├── requirements.txt

├── README.md

└── GlaucomaDetection.h5

---

## Installation

Clone the repository: git clone https://github.com/aymanmeethal/gaucoma-detection.git

```bash
git clone 
cd glaucoma-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Run:

```bash
python main.py train_model
```

Provide:

* Training dataset path
* Validation dataset path
* Number of epochs

The trained model will be stored as:

```text
GlaucomaDetection.h5
```

---

## Prediction

Run:

```bash
python main.py
```

Provide the path of:

* A single retinal image
* Or a folder containing multiple retinal images

Example Output:

```text
This image most likely belongs to Glaucoma_Positive with a 96.45% confidence.
```

---

## Applications

* Computer-aided glaucoma screening
* Ophthalmology research
* Clinical decision support systems
* Medical image analysis

---

## Future Enhancements

* Grad-CAM explainability visualization
* Multi-class retinal disease classification
* Web-based deployment using Flask
* Mobile application support
* Clinical report generation

---

## Author

Ayman Meethal
