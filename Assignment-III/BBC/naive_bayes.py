import numpy as np
import random
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import math
from feature_extracion import Tf_idf, split_data, BOW, only_Idf



tfidf = Tf_idf()
X, y, data = tfidf.preprocess_data()
X, y = tfidf.extract_top_k_features(X, y, 1500, data)
X_train, X_test, y_train, y_test = split_data(X, y)

bow = BOW()
X, y, data = bow.preprocess_data()
X, y = bow.extract_top_k_features(X, y, 150)
X_train, X_test, y_train, y_test = split_data(X, y)

idf = only_Idf()
X, y, data = idf.preprocess_data()
X, y = idf.extract_top_k_features(X, y, 1500, data)
X_train, X_test, y_train, y_test = split_data(X, y)


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

def NaiveBayes(train_images, train_labels, test_images, smoothing):
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
    prediction = NaiveBayes(X_train, np.array(y_train), X_test, 1)
    accuracy = calculate_accuracy(np.array(prediction), y_test)
    print("Accuracy:", accuracy)
    accuracies = []
    for laplace_smoothing in LAPLACE_SMOOTHING_VALUES:
        print(f'Training Naive Bayes with Laplace smoothing = {laplace_smoothing}...')
        prediction = NaiveBayes(X_train, np.array(y_train),  X_test, 1)
        accuracy = calculate_accuracy(np.array(prediction),  y_test)
        accuracies.append(accuracy)
        print(f'Accuracy: {accuracy:.4f}')
    print(accuracies)
    # Plot the accuracy change based on the hyperparameter variations
    plt.plot(LAPLACE_SMOOTHING_VALUES, accuracies)
    plt.xlabel('Laplace smoothing')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Laplace smoothing')
    plt.show()
    #################################################
LAPLACE_SMOOTHING_VALUES = [0.1, 0.5, 1.0, 10, 100,1000]

if __name__=="__main__":
    main()