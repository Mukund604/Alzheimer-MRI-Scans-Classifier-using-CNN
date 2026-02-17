# Alzheimer's MRI Scans Classifier using CNN

![](https://www.frontiersin.org/files/Articles/501050/fnins-14-00259-HTML-r1/image_m/fnins-14-00259-g001.jpg)

A Convolutional Neural Network (CNN) that classifies Alzheimer's MRI scans into four categories: **Mild Demented**, **Moderate Demented**, **Non Demented**, and **Very Mild Demented**.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)

## Overview

Alzheimer's disease is a progressive neurodegenerative disorder that affects millions of people worldwide. Early and accurate diagnosis is crucial for effective management and treatment. This project utilizes deep learning to develop a classifier that can identify the stage of Alzheimer's disease from MRI scans across four classes.

## Dataset

The dataset is sourced from the [Alzheimer MRI Disease Classification Dataset](https://www.kaggle.com/datasets/fernando2rad/alzheimer-mri-disease-classification-dataset) on Kaggle. It contains MRI brain scans in parquet format, labeled into four classes:

| Label | Class |
|-------|-------|
| 0 | Mild Demented |
| 1 | Moderate Demented |
| 2 | Non Demented |
| 3 | Very Mild Demented |

- **Training samples:** 5,120
- **Test samples:** 1,280
- **Image size:** 128x128x3

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Mukund604/Alzheimer-MRI-Scans-Classifier-using-CNN.git
    cd Alzheimer-MRI-Scans-Classifier-using-CNN
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

The model uses a 4-block CNN with BatchNormalization, GlobalAveragePooling2D, and Dropout for regularization. Data augmentation (rotation, shift, flip, zoom) is applied during training to reduce overfitting.

```python
Model = Sequential()

# Block 1 - 32 filters
Model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)))
Model.add(BatchNormalization())
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Dropout(0.25))

# Block 2 - 64 filters
Model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
Model.add(BatchNormalization())
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Dropout(0.25))

# Block 3 - 128 filters
Model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
Model.add(BatchNormalization())
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Dropout(0.25))

# Block 4 - 256 filters
Model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
Model.add(BatchNormalization())
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Dropout(0.25))

# Classification head
Model.add(GlobalAveragePooling2D())
Model.add(Dense(256, activation='relu'))
Model.add(Dropout(0.5))
Model.add(Dense(4, activation='softmax'))
```

Training uses `EarlyStopping` and `ReduceLROnPlateau` callbacks.

## Results

The model is evaluated with a full classification report (precision, recall, F1-score per class) and confusion matrix on the held-out test set. See the notebook for detailed results after training.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to create an issue or submit a pull request.

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request
