# MNIST Classification Project

This repository contains my implementation of classification models on the MNIST dataset, exploring various feature extraction techniques and analyzing their impact on accuracy. The project aims to evaluate the performance of different machine learning algorithms on the MNIST dataset, leveraging key features to optimize classification accuracy.

## Features and Techniques

1. **Raw Pixel Intensity**: Utilizes the pixel values directly as features for classification.
2. **Histogram of Oriented Gradients (HOG)**: Extracts edge and shape information from images, making it effective for object recognition tasks.
3. **Principal Component Analysis (PCA)**: Reduces the dimensionality of the image data while preserving essential information, enhancing computational efficiency.

## Machine Learning Algorithms

- **Naive Bayes**: Evaluated with different smoothing parameters to determine the optimal performance.
- **Logistic Regression**: Tested with various learning rates to find the best accuracy on the MNIST dataset.

## Accuracy Analysis

The project includes a comprehensive analysis of how different feature extraction methods and model parameters affect classification accuracy. Graphs are provided to compare the performance across different settings.

## Technologies Used

- **Python**: For implementing the models and data processing.
- **Scikit-learn**: Utilized for machine learning algorithms and feature extraction methods.
- **Matplotlib/Seaborn**: Used for plotting accuracy analysis graphs.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/Blacknahil/ML-project
    ```
2. Navigate to the project directory:
    ```bash
    cd Assignment-III/Mnist-implementation
    ```
3. Run the Python scripts provided to see the implementation and analysis.
   ```bash
    python3 main.py
    ```

## Conclusion

This project provides a detailed exploration of feature selection techniques and their impact on machine learning models for the MNIST dataset. It highlights the importance of choosing the right features and parameters to achieve optimal classification accuracy.

