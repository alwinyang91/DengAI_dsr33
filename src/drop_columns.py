

class ColumnDropperTransformer:

    def __init__(self, columns):
        self.columns=columns

    def fit(self, X, y=None):
        return self 

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)
