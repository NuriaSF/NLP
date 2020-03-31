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

import pickle


class TfIdfVectorizer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self,
                 min_word_counts=1,
                 min_df=1, #min number of words (int)
                 max_df=1.0, #max percentage of words (float in [0,1])
                 doc_cleaner_pattern=r"('\w+)", #pattern for cleaning document
                 token_pattern='(?u)\\b\\w\\w+\\b', #pattern defining what a token is
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
        self.idf = []
        self.X_w = defaultdict(int)


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
        assert self.vocabulary == set(), "self.vocabulary is not empty it has {} words".format(len(self.vocabulary))
        assert isinstance(X,list), "X is expected to be a list of documents"

        self.word_to_ind = collections.OrderedDict() #vocab dictionary

        self.preprocessor.fit()

        for x in X: #X is the whole set of documents
            tokens = self.preprocessor.transform(x)
            ngrams = self._create_ngrams(tokens)

            for ngram in ngrams:
                if ngram not in self.word_to_ind.keys(): #if token is not yet in the vocab dictionary, add it
                    self.word_to_ind[ngram] = len(self.word_to_ind)
                self.X_w[ngram] += 1

        self._compute_idf(len(X))
        self._limit_features(self.transform(X))
        self._compute_idf(len(X)) #ugly but works
        

        self.n_features = len(self.word_to_ind)
        self.vocabulary = set(self.word_to_ind.keys())
        return self

    def fit_transform(self, X, y=None):
        self.fit(X,y)
        encoded_X = self.transform(X)
        return encoded_X

    def transform(self, X):

        mat = self._term_frequency(X)

        tfidf = mat.multiply(self.idf)
        norms = sparse.linalg.norm(tfidf, axis=1)
        norms[norms==0] = 1
        tfidf_norm = np.repeat(1/norms, mat.getnnz(axis=1))
        r,c = mat.nonzero()
        tfidf_norm = sparse.csr_matrix((tfidf_norm, (r,c)), shape=(mat.shape))
        tfidf = tfidf.multiply(tfidf_norm)

        return tfidf

    def _create_ngrams(self, tokens):
        min_n, max_n = self.ngram_range

        new_ngrams = []

        #ngrams
        for n in np.arange(min_n, min(max_n+1, len(tokens)+1)):
            for i in range(len(tokens)):
                if i < len(tokens) - n + 1:
                    ngram = tokens[i:i+n]
                    ngram = ' '.join(ngram)
                    new_ngrams.append(ngram)

        return new_ngrams

    def _limit_features(self, X):
        n_doc = X.shape[0]
        max_doc_count = (self.max_df if isinstance(self.max_df, numbers.Integral) else self.max_df * n_doc)
        min_doc_count = (self.min_df if isinstance(self.min_df, numbers.Integral) else self.min_df * n_doc)
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
                    del self.X_w[term]

        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower min_df or a higher max_df.")

        return X[:,kept_indices]

    def _compute_idf(self, n_documents):
        n_features = len(self.word_to_ind)

        row = np.zeros(n_features)
        column = np.zeros(n_features)
        data = np.zeros(n_features)

        for i, w in enumerate(self.X_w.keys()):
            row[i] = 0
            column[i] = self.word_to_ind[w]
            data[i] = np.log(n_documents/(1 + self.X_w[w]))

        self.idf = sparse.csr_matrix((data,(row, column)), shape=(1,n_features))

    def _term_frequency(self, corpus):
        data = []
        row = []
        col = []

        for m, doc in enumerate(corpus):
            tokens = self.preprocessor.transform(doc)
            ngrams = self._create_ngrams(tokens)

            for ngram in ngrams:
                if ngram in self.word_to_ind.keys(): #if the word is not in the vocab, ignore it
                    row.append(m) #we are dealing with the m-th document
                    col.append(self.word_to_ind[ngram])
                    data.append(1)

        return scipy.sparse.csr_matrix((data, (row,col)), shape=(len(corpus), len(self.word_to_ind)))

    def _convert(self,o):
        print("Object:", o)
        print("Type of Object:", type(o))
        print("Type of np.int32:", np.int32)
        print("isinstance(o, np.int32):", isinstance(o, np.int32))
        if isinstance(o,  np.generic): return o.item()
        raise TypeError

    def _create_param_dict(self):
        param_dict = dict()
        param_dict['min_word_counts'] = self.min_word_counts
        param_dict['dtype'] = self.dtype
        param_dict['max_df'] = self.max_df #int/float
        param_dict['min_df'] = self.min_df #int/float
        param_dict['vocabulary'] = self.vocabulary #set->list
        param_dict['word_to_ind'] = self.word_to_ind #orderdedDict->json
        param_dict['ngram_range'] = self.ngram_range #int tuple

        #param_dict['preprocessor'] = pickle.dumps(self.preprocessor)
        param_dict['doc_cleaner_pattern'] = self.doc_cleaner_pattern #str
        param_dict['token_pattern'] = self.token_pattern #str
        param_dict['stop_words'] = self.stop_words #set->list
        param_dict['idf'] = self.idf
        param_dict['X_w'] = self.X_w
        param_dict['document_cleaner_func'] = self.document_cleaner_func
        param_dict['tokenizer_func'] = self.tokenizer_func
        param_dict['token_cleaner_func'] = self.token_cleaner_func

        return param_dict

    def dumps(self):
        param_dict = self._create_param_dict()
        return pickle.dumps(param_dict)

    def dump(self, filename):
        param_dict = self._create_param_dict()
        with open(filename, 'wb+') as f:
            pickle.dump(param_dict, f)

    def load(self, filename):
        param_dict = dict()
        with open(filename, 'rb+') as f:
            param_dict = pickle.load(f)

        self.min_word_counts = param_dict['min_word_counts']
        self.dtype = param_dict['dtype']
        self.max_df = param_dict['max_df']
        self.min_df = param_dict['min_df']
        self.vocabulary = param_dict['vocabulary']
        self.word_to_ind = param_dict['word_to_ind']
        self.ngram_range = param_dict['ngram_range']

        self.doc_cleaner_pattern = param_dict['doc_cleaner_pattern']
        self.token_pattern = param_dict['token_pattern']
        self.stop_words = param_dict['stop_words']
        self.idf = param_dict['idf']
        self.X_w = param_dict['X_w']
        self.document_cleaner_func = param_dict['document_cleaner_func']
        self.tokenizer_func = param_dict['tokenizer_func']
        self.token_cleaner_func = param_dict['token_cleaner_func']

        self.preprocessor = Preprocessor(doc_cleaner_pattern=self.doc_cleaner_pattern, 
                                 token_pattern=self.token_pattern,
                                 document_cleaner_func=self.document_cleaner_func,
                                 tokenizer_func=self.tokenizer_func,
                                 token_cleaner_func=self.token_cleaner_func,
                                 stop_words=self.stop_words)
        self.preprocessor.fit()
