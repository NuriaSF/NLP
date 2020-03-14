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

    def term_frequency(self, corpus):
        n_features = len(self.vocabulary)
        n_documents = len(corpus)
        
        mat = sparse.lil_matrix((n_documents, n_features))

        for i, doc in enumerate(corpus):
            words = self.splitter.findall(doc.lower())
            for w in words:
                if w in self.vocabulary.keys():
                    mat[i,self.vocabulary[w]]+=1
        
        return mat.tocsr()

    def compute_idf(self, n_documents):
        n_features = len(self.vocabulary)
        idf = np.zeros([1, n_features])

        for w in self.X_w:
            idf[0][self.vocabulary[w]] = np.log(n_documents/(1 + self.X_w[w]))

        return sparse.csr_matrix(idf)
    
    def transform(self, X):
        idf = self.compute_idf(len(X))
        mat = self.term_frequency(X)
        
        tfidf = mat.multiply(idf)
        tfidf = tfidf/sparse.linalg.norm(tfidf)
        return tfidf
    
    def fit_transform(self, X):
        self.fit(X)
        mat = self.transform(X)
        return mat