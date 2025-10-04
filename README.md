# Deep Learning for Computer Vision Projects

**Author:** Grâce Esther DONG  
**Academic Program:** 4th Year Engineering - AI Specialization  
**Institution:** Aivancity School for Technology, Business & Society  
**Academic Year:** 2024-2025

## Project Overview

This repository contains multiple computer vision projects implementing advanced deep learning techniques including GANs, image style transfer, and cartoonization algorithms.

## Projects Included

### 1. Image Cartoonization
- **Description:** Convert real images into cartoon-style artwork using computer vision techniques
- **Techniques Used:** 
  - Gaussian and median filtering
  - Bilateral filtering for noise reduction
  - Laplacian edge detection
  - Color quantization
  - Mask combination
- **File:** `cartoonize.ipynb`

### 2. Pix2Pix Implementation
- **Description:** Conditional GAN for image-to-image translation
- **Architecture:** Generator-Discriminator network
- **Applications:** Style transfer, image enhancement
- **File:** `pix2pix.ipynb`

### 3. GAN-based Style Transfer
- **Description:** Generative models for artistic style transfer
- **Models:** Various generator architectures trained on different datasets
- **Outputs:** Multiple style variations and artistic transformations

## Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Computer vision operations
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization
- **scikit-learn** - Machine learning utilities

## Repository Structure

```
Deep-Learning-Computer-Vision-Projects/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── cartoonize.ipynb
│   ├── pix2pix.ipynb
│   └── style_transfer.ipynb
├── models/
│   ├── generator.keras
│   ├── generator_cars.keras
│   └── gan_model.keras
├── data/
│   └── sample_images/
├── results/
│   └── generated_samples/
└── utils/
    └── image_processing.py
```

## Key Features

- **Real-time Image Processing:** Efficient algorithms for live image transformation
- **Multiple Style Options:** Various artistic styles and cartoon effects
- **Pre-trained Models:** Ready-to-use generators for different image types
- **Comprehensive Pipeline:** End-to-end solution from preprocessing to final output

## Results

The projects demonstrate successful implementation of:
- High-quality image cartoonization with preserved details
- Effective style transfer maintaining content structure
- GAN training with stable convergence
- Multiple artistic style variations

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Usage
```python
# Load and run cartoonization
from utils.image_processing import generate_cartoon_image
generate_cartoon_image('input_image.jpg', 'cartoon_output.jpg')
```

## Academic Context

These projects were developed as part of the **Deep Learning for Computer Vision** course, focusing on:
- Advanced neural network architectures
- Generative modeling techniques
- Computer vision applications
- Real-world problem solving in image processing

## Achievements

- Successful implementation of multiple GAN architectures
- High-quality artistic style transfer results
- Efficient real-time image processing pipeline
- Comprehensive understanding of computer vision techniques

## License

This project is developed for academic purposes. Please cite appropriately if used for research.

## Contact

**Grâce Esther DONG**

---
*Developed with passion for computer vision and deep learning*