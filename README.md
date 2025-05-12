# Multi-Layer Perceptron (MLP) for Classification

This project implements a Multi-Layer Perceptron (MLP) from scratch using Python and NumPy for classification tasks. It includes data loading, preprocessing, model definition, training, and evaluation.

## Overview

The MLP is designed to classify data into two categories. The implementation includes:

-   **Data Loading and Preprocessing**:  Reading data from Excel files and normalizing the features.
-   **Model Architecture**: A two-layer neural network with a configurable number of hidden units.
-   **Activation Functions**: Sigmoid for the hidden layer and Softmax for the output layer.
-   **Training**:  Utilizing forward propagation, backpropagation, and gradient descent to update network weights.
-   **Evaluation**:  Calculating loss, accuracy, and generating a confusion matrix to assess performance.

## Files

-   `Assignment__002_A.ipynb`:  Jupyter Notebook containing the Python implementation of the MLP.
-   `DataSets/THA2train.xlsx`: Excel file containing the training dataset.
-   `DataSets/THA2validate.xlsx`: Excel file containing the validation dataset.

## Dependencies

-   Python 3.x
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn

## Setup

1.  Ensure you have Python 3.x installed.
2.  Install the required libraries using pip:

    ```bash
    pip install numpy pandas matplotlib seaborn
    ```

3.  Place the datasets (`THA2train.xlsx`, `THA2validate.xlsx`) in the `DataSets/` directory.

## Usage

1.  Open and run the `Assignment__002_A.ipynb` notebook in a Jupyter environment.
2.  The notebook will execute the data loading, model training, and evaluation steps.
3.  The results, including training/validation loss and the confusion matrix, will be displayed in the notebook.

## Methodology

### 1. Data Loading and Preprocessing

-   The training and validation datasets are loaded from Excel files using Pandas.
-   Features (X) and labels (y) are separated. Labels are one-hot encoded.
-   The features are normalized using z-score normalization to improve training performance.

### 2. Model Architecture

-   The MLP class defines the neural network:
    -   `__init__`: Initializes the weights and biases with random values. Weights are initialized using a method that takes into account the size of the previous and next layers to stabilize training.
    -   `forward`:  Performs forward propagation through the network.
    -   `backward`: Implements the backpropagation algorithm to compute gradients.
    -   `update_weights`: Updates the model's weights and biases based on the calculated gradients and the learning rate.
    -   `compute_loss`: Calculates the categorical cross-entropy loss.

### 3. Activation Functions

-   `sigmoid`:  Applies the sigmoid function, optimized to prevent overflow.
-   `sigmoid_derivative`: Computes the derivative of the sigmoid function.
-   `softmax`:  Applies the softmax function to the output layer to obtain probability distributions for each class.

### 4. Training

-   The model is trained using a batch gradient descent approach.
-   The training loop iterates over the dataset for a specified number of epochs.
-   In each epoch, the dataset is shuffled, and it is divided into batches.
-   For each batch, forward propagation is used to generate predictions, backpropagation calculates the gradients, and the weights are updated.
-   Training and validation loss are computed and recorded for each epoch to monitor training progress.

### 5. Evaluation

-   Accuracy is calculated on the validation set by comparing the predicted labels with the true labels.
-   A confusion matrix is generated to provide a detailed view of the classification performance.
-   Training and validation loss are plotted over epochs to visualize convergence and detect overfitting.

## Results

-   The notebook outputs the training and validation loss at specified epochs.
-   It also displays the validation accuracy and the confusion matrix after training.
-   Loss curves are plotted to analyze the training dynamics.

## Author

This project was developed as part of a Knowledge Discovery and Data Mining course.
