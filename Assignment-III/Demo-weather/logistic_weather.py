import numpy as np
import math
import weather_data
import matplotlib.pyplot as plt


class CustomLogisticRegression:
    def __init__(self, train_images, train_labels, learning_rate, num_iterations=100):
        self.train_images = train_images
        self.train_labels = train_labels
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.train(self.train_images, self.train_labels)

    def softmax(self, x):
        max_value = np.max(x)
        exp_scores = np.exp(x - max_value)
        softmax_probs = exp_scores / np.sum(exp_scores)
        return softmax_probs

    def dot_product(self, image, weight):
        return np.dot(image, weight)

    def train(self, x, y):
        feature_size = len(x[0])
        self.weights = np.zeros((2, feature_size))

        for _ in range(self.num_iterations):
            for index, image in enumerate(x):
                dot_products = np.dot(self.weights, image)
                softmax_scores = self.softmax(dot_products)

                targets = np.zeros(2)
                targets[int(y[index])] = 1

                errors = targets - softmax_scores
                weight_updates = self.learning_rate * np.outer(errors, image)
                self.weights += weight_updates

    def predict(self, image):
        dot_products = np.dot(self.weights, image)
        probabilities = self.softmax(dot_products)
        return np.argmax(probabilities)


def organize_data(data_set):
    labels = []
    data = []
    for label, samples in data_set.items():
        for sample in samples:
            data.append(sample)
            labels.append(label)

    return labels, data


data_set = weather_data.read_csv_file(".\weather.csv")
test_set = weather_data.read_csv_file(".\weather_test.csv")
train_labels, train_data = organize_data(data_set)
test_labels, test_data = organize_data(test_set)
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
accuracies = []
learning_rate_values = []
for learning_rate in learning_rates:
    model = CustomLogisticRegression(
        train_data, train_labels, learning_rate=learning_rate)
    count = 0
    for index, test_sample in enumerate(test_data):
        if model.predict(test_sample) == test_labels[index]:
            count += 1
    accuracy = count / len(test_data) * 100
    print("Learning Rate:", learning_rate)
    print("Accuracy:", accuracy, "%")
    print()
    accuracies.append(accuracy)
    learning_rate_values.append(learning_rate)

plt.plot(learning_rate_values, accuracies, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.title('Accuracy Change based on Learning Rate')
plt.show()
