from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

X, y = make_regression(100, 5)
X_df = pd.DataFrame(X)
y_df = pd.DataFrame(y)

pipe = make_pipeline(
    StandardScaler(),
    RandomForestRegressor()
)

param_grid = {
    "randomforestregressor__n_estimators": [10, 100],
    "randomforestregressor__min_samples_split": [2, 5]
}

gscv = GridSearchCV(pipe, param_grid=param_grid)

gscv.fit(X_df, y_df)
