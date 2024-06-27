from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import coo_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class BOW:
    def __init__(self, data_folder='./bbc/'):
        self.data_folder = data_folder

    def extract_top_k_features(self, X, y, k):
        vectorizer = CountVectorizer()
        feature_names = []
        feature_scores = np.asarray(X.sum(axis=0)).ravel()
        top_k_indices = feature_scores.argsort()[::-1][:k]
        return X[:, top_k_indices], y


    def preprocess_data(self):
        # Load the data
        with open(self.data_folder + 'bbc.classes', 'r') as f:
            classes = f.read().splitlines()[4:]

        with open(self.data_folder + 'bbc.docs', 'r') as f:
            docs = f.read().splitlines()

        with open(self.data_folder + 'bbc.terms', 'r') as f:
            terms = f.read().splitlines()

        with open(self.data_folder + 'bbc.mtx', 'r') as f:
            matrix = f.readlines()[2:]  # skip the header lines

        corpus = [(int(doc)-1, int(term)-1, int(float(freq))) for term, doc, freq in (line.split() for line in matrix)]
        doc_ids, term_ids, freqs = zip(*corpus)
        X = np.zeros((len(docs), len(terms)))
        for doc_id, term_id, freq in corpus:
            X[doc_id, term_id] = freq
       
        y = [int(line.split()[1]) for line in classes]

        return X, y, docs
    


class Tf_idf:
    def __init__(self, data_folder='./bbc/'):
        self.data_folder = data_folder
    
    def extract_top_k_features(self, X, y, k, data):
        # Extract top k features for each class using TF-IDF
        feature_names = []
        vectorizer = TfidfVectorizer(norm=None, use_idf=True, preprocessor=self.extract_freq_count)
        X_class_tfidf = vectorizer.fit_transform(data)
        feature_scores = np.asarray(X_class_tfidf.sum(axis=0)).ravel()
        top_k_indices = feature_scores.argsort()[::-1][:k]
        feature_names.extend(vectorizer.get_feature_names_out()[top_k_indices])
        feature_names = list(set(feature_names))
        return X[:, top_k_indices], y

    def extract_freq_count(self, doc_tuple):
        return ' '.join([term + ' ' + str(freq) for term, freq in doc_tuple[2].items()])

    def preprocess_data(self):
        # Load the data
        with open(self.data_folder + 'bbc.classes', 'r') as f:
            classes = f.read().splitlines()[4:]

        with open(self.data_folder + 'bbc.docs', 'r') as f:
            docs = f.read().splitlines()

        with open(self.data_folder + 'bbc.terms', 'r') as f:
            terms = f.read().splitlines()

        with open(self.data_folder + 'bbc.mtx', 'r') as f:
            matrix = f.readlines()[2:]  # skip the header lines

        # Preprocess the data
        corpus_dict = {}
        for i, line in enumerate(matrix):
            term_id, doc_id, freq = line.split()
            doc_name = f"docs{doc_id}"
            term_idx = int(term_id) - 1  # convert to 0-based index
            term = terms[term_idx]
            freq = int(float(freq))
            if freq >= 0:
                if doc_name not in corpus_dict:
                    corpus_dict[doc_name] = {}
                corpus_dict[doc_name][term] = freq

        data = []
        for i, (doc_name, freq_dict) in enumerate(corpus_dict.items()):
            doc_index = i
            data.append((doc_name, doc_index, freq_dict))

        # Create a sparse matrix from the data
        corpus = [(int(doc)-1, int(term)-1, int(float(freq))) for term, doc, freq in (line.split() for line in matrix)]
        doc_ids, term_ids, freqs = zip(*corpus)
        X = np.zeros((len(docs), len(terms)))
        for doc_id, term_id, freq in corpus:
            X[doc_id, term_id] = freq

        y = [int(line.split()[1]) for line in classes]

        return X, y, data




class only_Idf:
    def __init__(self, data_folder='./bbc/'):
        self.data_folder = data_folder
    
    def extract_top_k_features(self, X, y, k, data):
        vectorizer = TfidfVectorizer(norm=None, use_idf=True, preprocessor=self.extract_freq_count, smooth_idf=False, sublinear_tf=False)
        X_transformed = vectorizer.fit_transform(data)
        feature_scores = np.asarray(vectorizer.idf_)
        top_k_indices = feature_scores.argsort()[::-1][:k]
        feature_names = np.array(vectorizer.get_feature_names_out())[top_k_indices]
        return X[:, top_k_indices], y

    def extract_freq_count(self, doc_tuple):
        # Ignore the term frequencies and only return the terms
        return ' '.join([term for term, _ in doc_tuple[2].items()])

    def preprocess_data(self):
        # Load the data
        with open(self.data_folder + 'bbc.classes', 'r') as f:
            classes = f.read().splitlines()[4:]

        with open(self.data_folder + 'bbc.docs', 'r') as f:
            docs = f.read().splitlines()

        with open(self.data_folder + 'bbc.terms', 'r') as f:
            terms = f.read().splitlines()

        with open(self.data_folder + 'bbc.mtx', 'r') as f:
            matrix = f.readlines()[2:]  # skip the header lines

        # Preprocess the data
        corpus_dict = {}
        for i, line in enumerate(matrix):
            term_id, doc_id, freq = line.split()
            doc_name = f"docs{doc_id}"
            term_idx = int(term_id) - 1  # convert to 0-based index
            term = terms[term_idx]
            freq = int(float(freq))
            if freq >= 0:
                if doc_name not in corpus_dict:
                    corpus_dict[doc_name] = {}
                corpus_dict[doc_name][term] = freq

        data = []
        for i, (doc_name, freq_dict) in enumerate(corpus_dict.items()):
            doc_index = i
            data.append((doc_name, doc_index, freq_dict))

        X = np.zeros((len(docs), len(terms)))
        for i, (doc_name, _, freq_dict) in enumerate(data):
            for term, freq in freq_dict.items():
                term_index = terms.index(term)
                X[i, term_index] = freq

        y = [int(line.split()[1]) for line in classes]

        return X, y, data



def split_data(X, y, test_size=0.2, random_state=42):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test












