import re
import numpy as np
from scipy import sparse
from collections import defaultdict


class TfIdfVectorizer():
    def __init__(self, splitter='(?u)\\b\\w\\w+\\b'):
        
        self.splitter = re.compile(splitter)
        self.X_w = defaultdict(int)
        self.vocabulary = dict()
        
    
    def fit(self, corpus):
        """
        This function has to return X_w, a dict containing for each key, how
        many documents having that key are in our corpus.
        """

        idx = 0
        for document in corpus:
            words      = set(self.splitter.findall(document.lower()))
            for word in words:
                if word not in self.vocabulary.keys():
                    self.vocabulary[word] = idx
                    idx += 1
                self.X_w[word] += 1

    def term_frequency(self, document):
        words = self.splitter.findall(document.lower())
        n_features = len(self.vocabulary)

        word_indices = []
        for w in words: 
            if w in self.vocabulary.keys(): #if the word is not in the vocabulary we ignore it
                word_indices.append(self.vocabulary[w])

        return sparse.csr_matrix( (np.ones(len(word_indices)), (np.zeros(len(word_indices)), word_indices)), shape=(1, n_features))

    def compute_idf(self, n_documents):
        n_features = len(self.vocabulary)
        idf = np.zeros([1, n_features])

        for w in self.X_w:
            idf[0][self.vocabulary[w]] = np.log(n_documents/(1 + self.X_w[w]))

        return sparse.csr_matrix(idf)
    
    def transform(self, X):
        idf = self.compute_idf(len(X))
        mat = None
        for doc in X:
            tf = self.term_frequency(doc)
            tfidf = tf.multiply(idf)
            tfidf = tfidf/sparse.linalg.norm(tfidf)
            if mat is None:
                mat = tfidf
            else:
                mat = sparse.vstack([mat, tfidf])
        return mat
    
    def fit_transform(self, X):
        self.fit(X)
        mat = self.transform(X)
        return mat