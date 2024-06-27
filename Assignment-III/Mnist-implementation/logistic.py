import numpy as np
from mnist import MNIST
from sklearn.decomposition import PCA
from skimage.feature import hog
import math
import random

class MnistLogisticRegression:
    def __init__(self, n_iterations, lr):
        self.n_iterations = n_iterations
        self.learning_rate = lr

    def Softmax(self, z):
        """
        Calculate the softmax value given the score.
        args:
            Z: score -> w*x
        return:
            softmax of z
        """
        exp = [math.exp(x - max(z)) for x in z]
        return [x / sum(exp) for x in exp]

    def OneHot(self, y, c):
        """
        Takes a list y of integer values and an integer c as input, 
        and returns a one-hot encoded representation of the input list.
        args:
            y:  A list of y of integer values.
            c: classes
        Return:
            one-hot encoded representation of the input list.
        """
        # Construct a zero matrix
        y_encoded = [[0 for _ in range(c)] for _ in range(len(y))]
        # Giving 1 for some columns
        for i in range(len(y)):
            y_encoded[i][y[i]] = 1
        return y_encoded

    def fit(self, X, y, c):
        """
        Train based on the training image data set and  update the weight and bias values.
        args:
            X: List of iamges for training.
            y: Lis of labels for the training images.
            c: No of classes.
        """
        m, n = len(X), len(X[0])

        self.weights = [[random.random() for _ in range(c)] for _ in range(n)]
        self.bias = [random.random() for _ in range(c)]

        loss_arr = []

        for epoch in range(self.n_iterations):
            z = [[sum(X[i][j] * self.weights[j][k] for j in range(n)) + self.bias[k] for k in range(c)] for i in range(m)]

            grad_for_w = [[(1 / m) * sum((self.Softmax(z[i])[k] - self.OneHot(y, c)[i][k]) * X[i][j] for i in range(m)) for k in range(c)] for j in range(n)]

            grad_for_b = [(1 / m) * sum(self.Softmax(z[i])[k] - self.OneHot(y, c)[i][k] for i in range(m)) for k in range(c)]

            # Make copies of weights and biases before the update
            weights_copy = [list(row) for row in self.weights]
            bias_copy = list(self.bias)

            for j in range(n):
                for k in range(c):
                    self.weights[j][k] -= self.learning_rate * grad_for_w[j][k]

            for k in range(c):
                self.bias[k] -= self.learning_rate * grad_for_b[k]

            loss = -sum(math.log(self.Softmax(z[i])[y[i]]) for i in range(m)) / m
            loss_arr.append(loss)
            print('Epoch: {}, Loss: {}'.format(epoch, loss))

            # Assign back the copies to weights and biases
            self.weights = weights_copy
            self.bias = bias_copy

        return self.weights, self.bias, loss_arr

    def predict(self, X):
        """
        Predicts the digit of the image for image in the image dataset.
        args:
            List of images.
        Return:
            List of prediction for each image.
        """
        m, n, c = len(X), len(X[0]), len(self.bias)
        z = [[sum(X[i][j] * self.weights[j][k] for j in range(n)) + self.bias[k] for k in range(c)] for i in range(m)]

        y_hat = [self.Softmax(row) for row in z]

        return [max(range(len(row)), key=row.__getitem__) for row in y_hat]

def pixel_intensity_feature(images):
    images = images/ 255
    return (images > 0.5).astype(float)

    
if __name__=="__main__":

    def load_mnist_data():
        mndata = MNIST('./mnist_dataset')
        train_images,  train_labels = mndata.load_training()
        test_images, test_labels = mndata.load_testing()
        return train_images, train_labels, test_images, test_labels

    X_train, y_train, x_test, y_test = load_mnist_data()
    X_train = pixel_intensity_feature(np.array(X_train))
    x_test = pixel_intensity_feature(np.array(x_test))
    logistic = MnistLogisticRegression(1500, 1.0)

    # Using the row pixel intensity only 
    w, b, loss = logistic.fit(np.array(X_train[:10000]), np.array(y_train[:10000]), c=10)
    predictions = logistic.predict(np.array(x_test))
    # accuracy = calculate_accuracy(predictions, np.array(y_test))
    # print(accuracy)

    # Using the hog feature

    # features = hog_feature(X_train)
    # w, b, loss = logistic.fit(np.array(features[:10000]), np.array(y_train[:10000]), c=10)
    # features_test = hog_feature(x_test)
    # predictions = logistic.predict(np.array(features_test))
    # accuracy = calculate_accuracy(predictions, np.array(y_test))
    # print(accuracy)

    # Using the pca feature

    # PCA_feature(X_train)
    # PCA_feature(x_test)
    # w, b, loss = logistic.fit(np.array(X_train[:10000]), np.array(y_train[:10000]), c=10)
    # predictions = logistic.predict(np.array(x_test))
    # accuracy = calculate_accuracy(np.array(predictions), np.array(y_test))
    # print(accuracy)