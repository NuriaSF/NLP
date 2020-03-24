import pandas as pd
import scipy
import sklearn
from sklearn import *
import numpy as np
import collections
import numbers
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
                 min_df=1, #min number of words (int)
                 max_df=1.0, #max percentage of words (float in [0,1])
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

        self.max_df = max_df
        self.min_df = min_df

        self.vocabulary = set() #set containing all words in our vocabulary
        self.word_to_ind = collections.OrderedDict() #dictionary of the vocabulary (key=word, value=integer)
        self.ngram_range = ngram_range

        self.preprocessor = Preprocessor(doc_cleaner_pattern=doc_cleaner_pattern, 
                                         token_pattern=token_pattern,
                                         document_cleaner_func=document_cleaner_func,
                                         tokenizer_func=tokenizer_func,
                                         token_cleaner_func=token_cleaner_func,
                                         stop_words=stop_words)

        self.doc_cleaner_pattern = doc_cleaner_pattern
        self.token_pattern = token_pattern
        self.document_cleaner_func = document_cleaner_func
        self.tokenizer_func = tokenizer_func
        self.token_cleaner_func = token_cleaner_func
        self.stop_words = stop_words

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        assert self.vocabulary == set(), "self.vocabulary is not empty it has {} words".format(len(self.vocabulary))
        assert isinstance(X,list), "X is expected to be a list of documents"
        assert len(X)%2 == 0, "There are unpaired sentences in X." #This is a small modification for the CountVectorizer to suit our problem.

        n_docs = len(X)

        word_to_ind = collections.OrderedDict() #vocab dictionary

        self.preprocessor.fit()

        for x in X: #X is the whole set of documents

            tokens = self.preprocessor.transform(x)

            #ngrams
            for n in np.arange(self.ngram_range[0], self.ngram_range[1]+1):
                for token in tokens:
                    inx = tokens.index(token)
                    ngram = None
                    if inx+n < len(tokens):
                        ngram = tokens[inx:inx+n]
                        ngram = ' '.join(ngram)
                    elif n==1: #In the case where we are adding 1-grams, last word of each sentence was left out. This is a dirty solution.
                        ngram = token

                    if (ngram is not None) and (ngram not in word_to_ind.keys()): #if token is not yet in the vocab dictionary, add it
                            word_to_ind[ngram] = len(word_to_ind)

        self.word_to_ind =  word_to_ind
        self.n_features = len(word_to_ind)
        self.vocabulary = set(word_to_ind.keys())

        Xq1 = self.transform(X[:int(n_docs/2)])
        Xq2 = self.transform(X[int(n_docs/2):])
        encoded_X = self._limit_features(Xq1, Xq2) #This does min_df and max_df properly considering our problem

        return encoded_X

    def _limit_features(self, Xq1, Xq2):
        n_doc = Xq1.shape[0]
        max_doc_count = (self.max_df if isinstance(self.max_df, numbers.Integral) else self.max_df * n_doc)
        min_doc_count = (self.min_df if isinstance(self.min_df, numbers.Integral) else self.min_df * n_doc)


        X = sparse.vstack([Xq1, Xq2], format='csr')
        dfs = np.bincount(X.indices, minlength=X.shape[1])
        mask = np.ones(len(dfs), dtype=bool)
        if max_doc_count is not None:
            mask &= dfs <= max_doc_count
        if min_doc_count is not None:
            mask &= dfs >= min_doc_count

        # Remove words from vocabulary & word_to_ind
        if(any(~mask)):
            new_indices = np.cumsum(mask) - 1
            #removed_terms = set()
            for term, old_index in list(self.word_to_ind.items()):
                if mask[old_index]:
                    self.word_to_ind[term] = new_indices[old_index]
                else:
                    del self.word_to_ind[term]
                    self.vocabulary.difference(set(term))
            #        removed_terms.add(term)
        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower min_df or a higher max_df.")

        Xq1 = Xq1[:,kept_indices]
        Xq2 = Xq2[:,kept_indices]
        X = sparse.hstack([Xq1, Xq2], format='csr')
        return X#, removed_terms

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
                    ngram = None
                    if inx+n < len(tokens):
                        ngram = tokens[inx:inx+n]
                        ngram = ' '.join(ngram)
                    elif n==1:
                        ngram = token

                    if (ngram is not None) and (ngram in self.word_to_ind.keys()): #if the word is not in the vocab, ignore it
                        ngram_index = self.word_to_ind[ngram]
                        row.append(m) #we are dealing with the m-th document
                        col.append(ngram_index)
                        data.append(1)

        encoded_X = scipy.sparse.csr_matrix((data, (row,col)), shape=(len(X), len(self.word_to_ind)))

        return encoded_X
