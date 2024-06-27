import numpy as np
import random
import pandas as pd
from mnist import MNIST
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from skimage.feature import hog
import math


def pixel_intensity_feature(images):
    """
    Normalizes the image pixels and assings 1 if the pixel value is greater than 0.5 and 0 otherwise.
    args:
        images: List of images pixels for training or testing.
    Returns:
        Normalized image pixel list
    """
    images = images/ 255
    return (images > 0.5).astype(float)

def hog_feature(images):
    """
    Extract hog featuers from image.
    args:
        images: List of images pixels for training or testing.
    Returns:
        Extracted features.
    """
    features = []
    for image in images:
        hog_features = hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2))
        features.append(hog_features)
    return np.array(features)

def load_mnist_data():
    """Loads the minist image data.
    Returns: 
        classified list of train images, train labels, test images and test labels
    """
    mndata = MNIST('./mnist_dataset')
    train_images,  train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    return train_images, train_labels, test_images, test_labels

def calculate_mean(images):
    """
    Calculate the mean of each image in the image dataset.
    args:
        images: List of images.
    Returns:
        List of calculated means.
    """
    total = [0] * len(images[0])
    for image in images:
        for i, pixel in enumerate(image):
            total[i] += pixel
    return [value / len(images) for value in total]

def calculate_variance(images, mean, smoothing):
    """
    Calculates the variance of each image pixel values.
    args:
        images: List of images.
        mean:   Calculated mean of each image.
        smoothing: smoothing factor for laplace smoothing.
    Returns:
        List of Calculated variances for each image.
    """
    total = [0] * len(images[0])
    for image in images:
        for i, pixel in enumerate(image):
            total[i] += (pixel - mean[i]) ** 2
    return [(value / len(images)) + smoothing for value in total]

def calculate_log_likelihood(image, mean, variance):
    """
    Calculate the log likelihood of a given image.
    args:
        image: List of pixels for the image.
        mean:  Calculated mean of the image pixel values.
        variance: Calcuated variance of the image pixel values.
    Returns:
        log_likelihood: Calculated value of log likelihood
    """
    log_likelihood = 0
    for pixel, mean_value, variance_value in zip(image, mean, variance):
        log_likelihood += (-0.5 * math.log(2 * math.pi * variance_value)) - (((pixel - mean_value) ** 2) / (2 * variance_value))
    return log_likelihood

def MnistNaiveBayes(train_images, train_labels, test_images, smoothing):
    """
    Predicts the digit of an image given the image pixel data.
    args:
        train_images: List of images used for training.
        train_labels: List of 
        test_images:  Lis of images used for testing.
        smoothing:    Smoothing factor for laplace smoothing.
    returns:
        predictions: Lis of predictions for each image. 
    """
    guassians = {}
    prior_probabs = {}
    label_train = set(train_labels)
    # For each class in training labels calculate the mean and variance of the class
    for c in label_train:
        current_X = [train_images[i] for i in range(len(train_labels)) if train_labels[i] == c]
        guassians[c] = {
            'mean': calculate_mean(current_X),
            'cov': calculate_variance(current_X, calculate_mean(current_X), smoothing)
        }

        #  Calculate and store the prior probablities of each class
        prior_probabs[c] = float(len([label for label in train_labels if label == c])) / len(label_train)
    
    # Predict the digit of the image
    N, D = len(test_images), len(test_images[0])
    probabilities = [[0] * len(guassians) for _ in range(N)]
    
    for i in range(N):
        for j, (c, g) in enumerate(guassians.items()):
            mean, cov = g['mean'], g['cov']
            probabilities[i][j] = calculate_log_likelihood(test_images[i], mean, cov) + math.log(prior_probabs[c])
    
    prediction = [max(range(len(guassians)), key=lambda x: probabilities[i][x]) for i in range(N)]
    
    return prediction


def calculate_accuracy(predictions, true_labels):
    """
    Calculate the accuracy of the given predictions.
    args:
        predictions: List of prediction by the algorithm.
        true_labels: Lis of the true digit value of the images.
    returns:
        accuracy: The accuracy of the prediction.
    """
    correct = np.sum(predictions == true_labels)
    total = len(predictions)
    accuracy = correct / total
    return accuracy












def main():
    train_images, train_labels, test_images, test_labels = load_mnist_data()

    # Normalize the input data and extract features
    
    features = pixel_intensity_feature(np.array(train_images))
    test_features = pixel_intensity_feature(np.array(test_images))
    # PCA_feature(train_images)
    features = hog_feature(features)
    test_features = hog_feature(test_features)
    
    # plt.imshow(train_images[0].reshape(28, 28))
    # plt.show()

    prediction = MnistNaiveBayes(features, np.array(train_labels), test_features, 0.1)
    # Calculate accuracy
    accuracy = calculate_accuracy(np.array(prediction), test_labels)
    print("Accuracy:", accuracy)
    #################################################

if __name__=="__main__":
    main()