# Alzheimer's MRI Scans Classifier

![](https://www.frontiersin.org/files/Articles/501050/fnins-14-00259-HTML-r1/image_m/fnins-14-00259-g001.jpg)

This project focuses on developing a Convolutional Neural Network (CNN) model to classify Alzheimer's disease from MRI scans. The model aims to differentiate between MRI scans of healthy individuals and those with Alzheimer's disease with high accuracy.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Alzheimer's disease is a progressive neurodegenerative disorder that affects millions of people worldwide. Early and accurate diagnosis is crucial for effective management and treatment. This project utilizes deep learning to develop a classifier that can accurately identify Alzheimer's disease from MRI scans.

## Dataset

The dataset consists of MRI scans labeled as either "Healthy" or "Alzheimer's". The images are preprocessed and divided into training, validation, and test sets to train and evaluate the model's performance.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/alzheimers-mri-classifier.git
    cd alzheimers-mri-classifier
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To train and evaluate the model, run the Jupyter notebook provided in the repository:

1. Start Jupyter Notebook:
    ```sh
    jupyter notebook
    ```

2. Open the `alzheimer-mri-scans-classifier-using-cnn-2.ipynb` notebook and run the cells sequentially.

## Model Architecture

The model uses a Convolutional Neural Network (CNN) architecture, which is effective for image classification tasks. The architecture includes several convolutional layers, pooling layers, and fully connected layers, with activation functions and dropout for regularization.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, num_channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## Results

The model was evaluated on the validation and test datasets, achieving the following results:

- **Validation Accuracy:** 93.82%
- **Test Accuracy:** 93.83%

The model demonstrated strong performance, correctly predicting a significant portion of the test samples.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to create an issue or submit a pull request.

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request
