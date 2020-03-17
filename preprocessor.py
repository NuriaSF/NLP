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

class Preprocessor():

	def __init__(self,
				 doc_cleaner_pattern=r"('\w+)|([^a-zA-Z0-9])", #pattern for cleaning document
				 token_pattern=r"(?u)\b\w+\b", #pattern defining what a token is
				 document_cleaner_func=None,
				 tokenizer_func=None,
				 token_cleaner_func=None,
				 stop_words=[]):

		self._retype = type(re.compile('hello, world'))

		self.doc_cleaner_pattern = doc_cleaner_pattern
		self.token_pattern       = token_pattern #definition of what a token is

		self.document_cleaner_func      = document_cleaner_func #function to perform the document cleaning
		self.tokenizer_func        = tokenizer_func #function to split the document into tokens
		self.token_cleaner_func = token_cleaner_func #function to perform the cleaning of the tokens

		self.stop_words = set(stop_words) #set of stop words


		self.doc_cleaner      = None
		self.doc_tokenizer    = None
		self.word_transformer = None



	def _document_cleaner(self, lower=True):
		"""
		By default, removes all the non alphanumeric characters along with any
		character that follows an apostrophe
		"""

		if self.document_cleaner_func: #inputted one
			return self.document_cleaner_func

		else: #default
			clean_doc_pattern = re.compile(self.doc_cleaner_pattern)
			if lower:
				return lambda doc: clean_doc_pattern.sub(" ", doc).lower()
			else:
				return lambda doc: clean_doc_pattern.sub(" ", doc)

	def _tokenizer(self):
		"""
		By default, the tokens will be the sets of alphanumeric characters
		separated by white spaces.
		Notice that a token may be composed of a single character.
		"""

		if self.tokenizer_func: #inputted one
			return self.tokenizer_func

		else: #default
			token_pattern_aux = re.compile(self.token_pattern)
			return lambda doc: token_pattern_aux.findall(doc)


	def _token_cleaner(self):

		if self.token_cleaner_func: #inputted one
			return self.token_cleaner_func
		else: #default
			return lambda word: word #identity function

	def fit(self):
		self.doc_cleaner      = self._document_cleaner()
		self.doc_tokenizer    = self._tokenizer()
		self.word_transformer = self._token_cleaner()

	def transform(self, doc):

		x = self.doc_cleaner(doc)
		tokens = self.doc_tokenizer(x)
		tokens_aux = []
		for w in tokens:
			tokens_aux.append(self.word_transformer(w))
		tokens = tokens_aux
		tokens = [tok for tok in tokens if tok not in self.stop_words] #remove stopping words

		return tokens
