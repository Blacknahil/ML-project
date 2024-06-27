import numpy as np
# from mnist import MNIST
# from sklearn.decomposition import PCA
# from skimage.feature import hog
import math
import random
from feature_extracion import Tf_idf, split_data, BOW,only_Idf
from logistic import MnistLogisticRegression

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


def  read_bbc_data():
    with open('bbc/bbc.classes') as f:
        train_labels_freq = {}
        document_class = {}
        for line in f.readlines()[4:]:
            document_class[int(line.strip().split()[0]) + 1] = line.strip().split()[-1]
            if line.split()[-1] not in train_labels_freq.keys():
                train_labels_freq[line.strip().split()[-1]] = 1
            else:
                train_labels_freq[line.strip().split()[-1]] += 1

    with open('bbc/bbc.terms') as f:
        vocab = [term.strip() for term in f.readlines()]

    with open('bbc/bbc.mtx') as f:
        lines = f.readlines()
        # print(lines[2].split())
        data = [[int(float(x)) for x in line.split()] for line in lines[2:]]

    word_in_doc = {}


    for line in data:
        if line[1] not in word_in_doc.keys():
            word_in_doc[line[1]] = [(line[0],line[2])]
        else:
            word_in_doc[line[1]].append((line[0],line[2]))

    # print(word_in_doc)
    train_final_data = {}
    train_data = []
    train_label = []
    for key in list(word_in_doc.keys())[:int(len(list(word_in_doc.keys())) * 0.9)]:
        docc = [0] * (len(vocab) + 1)
        for doc in word_in_doc[key]:
            # print(doc[0])
            docc[doc[0]] = doc[1]

        train_final_data[key] = docc
        # docc.sort()
        # docc.reverse()
        train_data.append(docc[:1500])
        train_label.append(int(document_class[key]))
  

    test_data = []
    test_label = []
    for key in list(word_in_doc.keys())[int(len(list(word_in_doc.keys())) * 0.9):]:
        docc = [0] * (len(vocab) + 1)
        for doc in word_in_doc[key]:
            # print(doc[0])
            docc[doc[0]] = doc[1]
        # docc.sort()
        # docc.reverse()
        test_data.append(docc[:1500])
        test_label.append(int(document_class[key]))
    return train_data,train_label, test_data, test_label

def bow(doc_freq):
    # create a vocabulary of unique words
    vocab = set()
    for doc in doc_freq:
        vocab.update(set(doc))
    vocab = sorted(vocab)
    
    # create a histogram of word occurrences for each document
    X = []
    for doc in doc_freq:
        histogram = [0] * len(vocab)
        for i, word in enumerate(vocab):
            if word in doc:
                histogram[i] = doc.count(word)
        X.append(histogram)
    
    return X

def tf_idf(doc_freq):
    # create a vocabulary of unique words
    vocab = set()
    for doc in doc_freq:
        vocab.update(set(doc))
    vocab = sorted(vocab)
    
    # compute the term frequency for each document
    tf = []
    for doc in doc_freq:
        tf_doc = [count / sum(doc) for count in doc]
        tf.append(tf_doc)
    
    # compute the inverse document frequency for each word
    idf = []
    for i, word in enumerate(vocab):
        df = sum(1 for doc in doc_freq if word in doc)
        idf_word = math.log(len(doc_freq) / df)
        idf.append(idf_word)
    
    # compute the tf-idf for each document
    X = []
    for i, doc in enumerate(tf):
        tf_idf_doc = [tf[i][j] * idf[j] for j in range(len(vocab))]
        X.append(tf_idf_doc)
    
    return X
# datas = read_bbc_data()
train_feaure = X_train
test_feature = X_test


learning_rate =  [0.0001, 0.001, 0.01, 0.1, 1.0, 1.5]

for rate in learning_rate:
    test = MnistLogisticRegression(1000,rate)

    test.fit(np.array(train_feaure),np.array(y_train),5)
    prediction = test.predict(np.array(test_feature))
    print(prediction)

    def calculate_accuracy(predictions, true_labels):
        correct = np.sum(predictions == true_labels)
        total = len(predictions)
        accuracy = correct / total
        return accuracy
    accuracy = calculate_accuracy(prediction,np.array(y_test))
    print(accuracy)