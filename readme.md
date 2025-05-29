# PyTorch Malimg Classifier

A simple PyTorch implementation of a Convolutional Neural Network (CNN) for classifying malware families based on their image representations. This project utilizes the Malimg dataset and is structured as a Google Colab notebook.

## Overview

The script performs the following key operations:
1.  **Dataset Acquisition:** Downloads the "Malimg" dataset (malware images) using the Kaggle API via `kagglehub`.
2.  **Data Preprocessing:**
    *   Loads images, converts them to grayscale, and resizes them to 64x64 pixels.
    *   Normalizes pixel values.
    *   Uses `LabelEncoder` to convert class names into numerical labels.
3.  **Model Definition:** Implements a simple CNN architecture with:
    *   Two convolutional layers with ReLU activation and max-pooling.
    *   Two fully-connected layers.
4.  **Data Handling:** Uses PyTorch `Dataset` and `DataLoader` for efficient data loading and batching during training and testing.
5.  **Training:** Trains the CNN model using Cross-Entropy loss and the Adam optimizer.
6.  **Evaluation:** Evaluates the trained model on a test set and reports the classification accuracy.

## Dataset

This project uses the **Malimg Dataset**.
*   **Source:** [Malimg Dataset on Kaggle](https://www.kaggle.com/datasets/manmandes/malimg)
*   The dataset consists of 9,339 malware samples belonging to 25 different families, where each sample is represented as a grayscale image.

## Requirements

*   Python 3.x
*   PyTorch
*   scikit-learn
*   Pillow (PIL)
*   NumPy
*   Kaggle (`kagglehub`)
*   `google-colab` (if running in Google Colab)

## Setup and Usage

This script is designed to be run in a Google Colab environment.

1.  **Kaggle API Token:**
    *   You will need a Kaggle API token (`kaggle.json`).
    *   Go to your Kaggle account page, then 'Account' tab, and click 'Create New API Token'. This will download `kaggle.json`.
2.  **Open in Colab:**
    *   Upload the `.ipynb` file (containing this code) to your Google Drive and open it with Google Colab.
    *   Alternatively, you can upload it directly to Colab.
3.  **Run the Notebook:**
    *   Execute the cells sequentially.
    *   When prompted by the `files.upload()` cell, upload your `kaggle.json` file.
    *   The script will then:
        *   Set up the Kaggle API credentials.
        *   Download the Malimg dataset using `kagglehub`.
        *   Prepare the datasets and dataloaders.
        *   Define, train, and evaluate the CNN model.

## Code Structure (within the Notebook)

The notebook is organized into the following main sections:

*   **Importing Libraries:** Imports all necessary Python packages.
*   **Uploading File And Creating Dataset:**
    *   Handles the upload of `kaggle.json`.
    *   Sets up Kaggle credentials.
    *   Downloads the Malimg dataset.
*   **Creating Model & Dataloader:**
    *   Defines the `SimpleCNN` model architecture.
    *   Defines the `MalimgDataset` class for custom data loading and preprocessing.
    *   Creates `DataLoader` instances for training and testing.
    *   Initializes the model, loss function (criterion), and optimizer.
*   **Training & Test & Evaluation:**
    *   Contains the training loop for the model.
    *   Includes the testing loop and accuracy calculation.

## Expected Output

The script will:
*   Print the path where the dataset is downloaded.
*   Print the training loss after each epoch.
*   Print the final test accuracy of the model on the Malimg test set.

## License

This project is open-source and available under the [Apache License 2.0](LICENSE).
