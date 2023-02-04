from sklearn.base import BaseEstimator

class ModelClass(BaseEstimator):

    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
