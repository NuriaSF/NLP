from sklearn.pipeline import Pipeline
import extra_features
import xgboost as xgb
from scipy import sparse
import sklearn
from CountVectorizer_BagOfWords import CountVectorizer as cv
from nltk.corpus import stopwords 

class CountVectorizerTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self,**params):
        self.CountVectorizer = cv()
        self.CountVectorizer.set_params(**params)
        
    def fit(self, X, y=None):
        self.CountVectorizer.fit(X)
        
    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X)
    
    def transform(self, X):
        X_tr = self.CountVectorizer.transform(X)
        nexamples, nvars = X_tr.shape
        split = (int)(nexamples/2)
        XX1 = X_tr[:split,:]
        XX2 = X_tr[split:,:]
        
        q1 = X[:split]
        q2 = X[split:]

        new_ft = self.compute_new_features(XX1, XX2, q1, q2)
        
        stack = [XX1, XX2]
        [stack.append(x) for k,x in new_ft.items()]
        
        return sparse.hstack(stack, format='csr')
    
    def compute_new_features(self, Xq1, Xq2, q1, q2):
        new_ft = {}
        new_ft['num1'], which1 = extra_features.has_numbers(q1)
        new_ft['num2'], which2 = extra_features.has_numbers(q2)
        new_ft['diffNum'] = extra_features.is_different_number(which1, which2)
        
        new_ft['math1'] = extra_features.is_math(q1)
        new_ft['math2'] = extra_features.is_math(q2)
        
        new_ft['len1'] = extra_features.get_qlength(q1)
        new_ft['len2'] = extra_features.get_qlength(q2)
        
        return new_ft
    
    def dump(self, filename):
        self.CountVectorizer.dump(filename)
    
    def load(self, filename):
        self.CountVectorizer.load(filename)

class XGBModel(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, **params):
        self.num_iters = 5000
        self.params = params
        self.model = None
        
    def fit(self, X, y):
        d_train = xgb.DMatrix(X, label=y)
        self.model = xgb.train(self.params, d_train, self.num_iters, verbose_eval=10)
        
    def predict(self, X):
        d_pred = xgb.DMatrix(X)
        return self.model.predict(d_pred)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return sklearn.metrics.roc_auc_score(y, y_pred)
    
    def dump(self,filename):
        self.model.save_model(filename)
        
    def load(self, filename):
        self.model = xgb.Booster()
        self.model.load_model(filename)