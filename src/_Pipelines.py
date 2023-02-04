# Pipeline
from sklearn.pipeline import make_pipeline

# Scaler
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Metrics
from sklearn.metrics import mean_absolute_error


class Pipelines():

    def __init__(self):
        pass

    # Pipelines for Model fitting and predicting
    def lr_pipe(self, x, y):
        pipe = make_pipeline(
            StandardScaler(),
            LinearRegression()
        )
        return pipe

    def rf_pipe(self, x, y):
        pipe = make_pipeline(
            StandardScaler(),
            RandomForestRegressor()
        )
        return pipe

    def xgb_pipe(self, x, y):
        pipe = make_pipeline(
            StandardScaler(),
            XGBRegressor()
        )
        return pipe

    # Pipeline for evaluate
    def eval(self, y_val, y_predict, name=None):
        error = mean_absolute_error(y_val, y_predict)
        print(f'MAE ({name}): {error}') 
