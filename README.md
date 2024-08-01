# Deepfake Detection

## Overview

Detects fake images using deep learning and forensic techniques to ensure authenticity in digital media.

## Methodology

- **Error Level Analysis (ELA):** Identifies tampering by comparing image compression levels.
- **Convolutional Neural Networks (CNNs):** Classifies images as real or fake by detecting anomalies.
- **Architecture:**  
  - Two CNN layers with ReLU activation  
  - Max-pooling and dropout layers  
  - Fully connected and output layers  
  - Optimized with RMSprop

## Results

- **Dataset:** CASIA (5500 training, 1300 testing images)
- **Validation Accuracy:** 93.53%

## Conclusion

Combining CNN and ELA effectively detects image manipulation, providing reliable authenticity verification.
