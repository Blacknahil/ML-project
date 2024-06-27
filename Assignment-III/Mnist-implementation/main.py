from mnist import MNIST
import numpy as np
from sklearn.decomposition import PCA
from skimage.feature import hog
from logistic import MnistLogisticRegression
from naive_bayes import MnistNaiveBayes

def pixel_intensity_feature(images):
    images = images/ 255
    return (images > 0.5).astype(float)


def hog_feature(images):
    features = []
    for image in images:
        hog_features = hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2))
        features.append(hog_features)
    return np.array(features)

def PCA_feature(images):
    pca = PCA(n_components=100)
    pca.fit(images)
    pca.transform(images[:1])
    np.dot(images[:1] - images.mean(axis=0), pca.components_.T) 
    return images

def calculate_accuracy(predictions, true_labels):
    correct = np.sum(predictions == true_labels)
    total = len(predictions)
    accuracy = correct / total
    return accuracy


def load_mnist_data():
    mndata = MNIST('./mnist_dataset')
    train_images,  train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()
    return train_images, train_labels, test_images, test_labels

if __name__=="__main__":

    X_train, y_train, x_test, y_test = load_mnist_data()
    X_train = pixel_intensity_feature(np.array(X_train))
    x_test = pixel_intensity_feature(np.array(x_test))


    # logistic = LogisticRegression(1500, 1.0)

    # Using the row pixel intensity only 
    # w, b, loss = logistic.fit(np.array(X_train[:10000]), np.array(y_train[:10000]), c=10)
    # prediction = logistic.predict(np.array(x_test))


    # Using the hog feature

    # features = hog_feature(X_train)
    # prediction = NaiveBayes(features, np.array(y_train), x_test)
    # Logistic regression 
    # w, b, loss = logistic.fit(np.array(features[:10000]), np.array(y_train[:10000]), c=10)
    # features_test = hog_feature(x_test)
    # prediction = logistic.predict(np.array(features_test))



    # Using the pca feature

    feature = PCA_feature(X_train)
    print("pca feature",feature)
    # PCA_feature(x_test)
    # w, b, loss = logistic.fit(np.array(X_train[:10000]), np.array(y_train[:10000]), c=10)
    # prediction = logistic.predict(np.array(x_test))

    # accuracy = calculate_accuracy(prediction, np.array(y_test))
    # print(accuracy)

    ####################Testing Naive Baye's algorithm on different smoothing values and on each feature extraction method########################
    smoothing = [0.1, 0.5, 1.0, 10, 100, 1000]
    feature_extractions = [pixel_intensity_feature, PCA_feature, hog_feature]
    accuracy_levles = []
    for smoothing_factor in smoothing:
        feature_extractions = [pixel_intensity_feature, PCA_feature, hog_feature]
        smoothing_accuracy = []
        for feature in feature_extractions:
            feature_accuracy = 0
            if feature == pixel_intensity_feature:
                prediction = MnistNaiveBayes(features, np.array(y_train), x_test, smoothing)
                accuracy = calculate_accuracy(prediction, np.array(y_test))
                feature_accuracy += accuracy
            if feature == PCA:
                PCA_feature(X_train)
                PCA_feature(x_test)
                prediction = MnistNaiveBayes(features, np.array(y_train), x_test, smoothing)
                accuracy = calculate_accuracy(prediction, np.array(y_test))
                feature_accuracy += accuracy
            if feature == hog:
                features = hog_feature(X_train)
                prediction = MnistNaiveBayes(features, np.array(y_train), x_test, smoothing)
                accuracy = calculate_accuracy(prediction, np.array(y_test))
                feature_accuracy += accuracy
        
            smoothing_accuracy.append(feature_accuracy)
        accuracy_levles.append(smoothing_accuracy)

    print(accuracy_levles)

    #############Testing Logistic Regression Algorithm with diffrent 
    ########learning rate and on each feature extraction method#############

    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]
    feature_extractions = [pixel_intensity_feature, PCA_feature, hog_feature]
    accuracy_levles = []
    for lr in learning_rates:
        feature_extractions = [pixel_intensity_feature, PCA_feature, hog_feature]
        smoothing_accuracy = []
        for feature in feature_extractions:
            feature_accuracy = 0
            if feature == pixel_intensity_feature:
                logistic = MnistLogisticRegression(1500, lr)
                w, b, loss = logistic.fit(np.array(X_train[:10000]), np.array(y_train[:10000]), c=10)
                predictions = logistic.predict(np.array(x_test))
                accuracy = calculate_accuracy(prediction, np.array(y_test))
                feature_accuracy += accuracy
            if feature == PCA:
                PCA_feature(X_train)
                PCA_feature(x_test)
                logistic = MnistLogisticRegression(1500, lr)
                w, b, loss = logistic.fit(np.array(X_train[:10000]), np.array(y_train[:10000]), c=10)
                predictions = logistic.predict(np.array(x_test))
                accuracy = calculate_accuracy(prediction, np.array(y_test))
                feature_accuracy += accuracy
            if feature == hog:
                features = hog_feature(X_train)
                logistic = MnistLogisticRegression(1500, lr)
                w, b, loss = logistic.fit(np.array(X_train[:10000]), np.array(y_train[:10000]), c=10)
                predictions = logistic.predict(np.array(x_test))
                accuracy = calculate_accuracy(prediction, np.array(y_test))
                feature_accuracy += accuracy
        
            smoothing_accuracy.append(feature_accuracy)
        accuracy_levles.append(smoothing_accuracy)

    print(accuracy_levles)
