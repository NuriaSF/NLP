import pandas as pd
import scipy
import sklearn
from sklearn import *
import numpy as np
import collections
from scipy import sparse
import nltk
from collections import defaultdict
import re
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer, SnowballStemmer
from nltk.corpus import stopwords
from scipy.sparse import hstack
from preprocessor import Preprocessor


class CountVectorizer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self,
                 min_word_counts=1,
                 doc_cleaner_pattern=r"('\w+)|([^a-zA-Z0-9])", #pattern for cleaning document
                 token_pattern=r"(?u)\b\w+\b", #pattern defining what a token is
                 dtype=np.float32,
                 document_cleaner_func=None,
                 tokenizer_func=None,
                 token_cleaner_func=None,
                 stop_words=[],
                 ngram_range=(1, 1)):

        self.min_word_counts     = min_word_counts
        self.dtype               = dtype

        self.vocabulary = set() #set containing all words in our vocabulary
        self.word_to_ind = collections.OrderedDict() #dictionary of the vocabulary (key=word, value=integer)
        self.ngram_range = ngram_range

        self.preprocessor = Preprocessor(doc_cleaner_pattern=doc_cleaner_pattern, 
                                         token_pattern=token_pattern,
                                         document_cleaner_func=document_cleaner_func,
                                         tokenizer_func=tokenizer_func,
                                         token_cleaner_func=token_cleaner_func,
                                         stop_words=stop_words)

    def fit(self, X):

        assert self.vocabulary == set(), "self.vocabulary is not empty it has {} words".format(len(self.vocabulary))
        assert isinstance(X,list), "X is expected to be a list of documents"

        word_to_ind = collections.OrderedDict() #vocab dictionary

        self.preprocessor.fit()

        for x in X: #X is the whole set of documents

            tokens = self.preprocessor.transform(x)  

            #ngrams
            for n in np.arange(self.ngram_range[0], self.ngram_range[1]+1):
                for token in tokens:
                    inx = tokens.index(token)
                    if inx+n < len(tokens):
                        ngram = tokens[inx:inx+n]
                        ngram = ' '.join(ngram)

                        if ngram not in word_to_ind.keys(): #if token is not yet in the vocab dictionary, add it
                            word_to_ind[ngram] = len(word_to_ind)


        self.word_to_ind =  word_to_ind
        self.n_features = len(word_to_ind)
        self.vocabulary = set(word_to_ind.keys())

        return self



    def transform(self, X):

        data = []
        row = []
        col = []

        for m, doc in enumerate(X):
            tokens = self.preprocessor.transform(doc)

            #ngrams
            for n in np.arange(self.ngram_range[0], self.ngram_range[1]+1):
                for token in tokens:
                    inx = tokens.index(token)
                    if inx+n < len(tokens):
                        ngram = tokens[inx:inx+n]
                        ngram = ' '.join(ngram)

                        if ngram in self.word_to_ind.keys(): #if the word is not in the vocab, ignore it
                            ngram_index = self.word_to_ind[ngram]
                            row.append(m) #we are dealing with the m-th document
                            col.append(ngram_index)
                            data.append(1)

        encoded_X = scipy.sparse.csr_matrix((data, (row,col)), shape=(m+1,len(self.word_to_ind)))

        return encoded_X



    def fit_transform(self, X, y=None):
        self.fit(X)
        encoded_X = self.transform(X)
        return encoded_X
