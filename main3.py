
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from src.utils import data_loader, get_data_into_submission_format, cyclical_encoding
from src.model_class import ModelClass
from src.drop_columns import ColumnDropperTransformer
from src.ProcessData import ProcessingData
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
# %% Load the data
df_train, df_test = data_loader()
df_train_features = pd.read_csv("./data/dengue_features_train.csv")
df_train_features.loc[:, "type"] = "TRAIN"
df_label = pd.read_csv("./data/dengue_labels_train.csv")

df_test = pd.read_csv("./data/dengue_features_test.csv")
df_test.loc[:, "type"] = "TEST"

df_train_with_label = pd.concat([df_train_features, df_label.loc[:,'total_cases']], axis=1)


df_total_no_label = pd.concat([df_train_features, df_test], axis=0)

# Turn date column into month/ day
df_total_no_label.loc[:, "month"] = pd.to_datetime(df_total_no_label.loc[:, "week_start_date"]).dt.month
df_total_no_label.drop(columns="week_start_date", inplace=True)



# %% To do feature engineering
ProcessingData.duplicates_drop(df_train)
ProcessingData.fill_data(df_train, fillType ='ffill')
to_trim = ['ndvi_ne', 'ndvi_nw',
       'ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
       'station_min_temp_c', 'station_precip_mm']
ProcessingData.drop_outlier(df_train, df_test, to_trim)


# %% fed the data
# X = df_train.drop(columns=['total_cases'])

y = df_train.loc[:, 'total_cases']

# Turn into cyclical feature
def cyclical_encoding(data: pd.DataFrame, column: str) -> pd.DataFrame:
    new_column_name_sin = f"sin_{column}"
    new_column_name_cos = f"cos_{column}"

    data.loc[:, new_column_name_sin] = np.sin(2 * np.pi * data.loc[:, column] / max(data.loc[:, column]))
    data.loc[:, new_column_name_cos] = np.cos(2 * np.pi * data.loc[:, column] / max(data.loc[:, column]))

    data.drop(columns=column, inplace=True)

    return data

df_total_no_label = cyclical_encoding(df_total_no_label, "weekofyear")
df_total_no_label = cyclical_encoding(df_total_no_label, "month")

# df_total_no_label = df_total_no_label.shift(8)
# df_total_no_label.iloc[8:,:]


X = df_total_no_label.loc[df_total_no_label.loc[:,'type'] == 'TRAIN']

# X_train_try, X_test_try, y_train_try, y_test_try = train_test_split(X, y, test_size=0.33, random_state=42)

X_test = df_total_no_label.loc[df_total_no_label.loc[:,'type'] == 'TEST']


ProcessingData.fill_data(X, fillType ='ffill')
ProcessingData.fill_data(X_test, fillType ='ffill')
# X.dropna(axis=0, inplace=True)
# X_test.dropna(axis=0, inplace=True)

# %% Create pipeline
# pipe = make_pipeline(
#     ColumnDropperTransformer(["city", "week_start_date"]),
#     StandardScaler(),
#     SimpleImputer(),
#     ModelClass(RandomForestRegressor())
# )

pipe = make_pipeline(
    ColumnDropperTransformer(["city"]),
    StandardScaler(),
    SimpleImputer(),
    RandomForestRegressor()
)

# %% Param Grid

param_grid = {
    "randomforestregressor__n_estimators": [100, 200],
    "randomforestregressor__min_samples_split": [2, 20, 50]
}

# %% Gridsearch

gscv = GridSearchCV(pipe, param_grid, scoring="neg_mean_absolute_error")

# %% Split

sj_x_train = X.query("city=='sj'")
sj_y_train = df_train.query("city=='sj'").loc[:, "total_cases"]
iq_x_train = X.query("city=='iq'")
iq_y_train = df_train.query("city=='iq'").loc[:, "total_cases"]
ProcessingData.fill_data(X_test, fillType ='ffill')
sj_x_test = X_test.query("city=='sj'")
iq_x_test = X_test.query("city=='iq'")

sj_x_train = sj_x_train.drop(columns='type')                                
sj_y_train = sj_y_train.drop(columns='type')
iq_x_train = iq_x_train.drop(columns='type')
iq_y_train = iq_y_train.drop(columns='type')
sj_x_test = sj_x_test.drop(columns='type')
iq_x_test = iq_x_test.drop(columns='type')

sj_model = gscv.fit(sj_x_train, sj_y_train)
sj_best_model = gscv.best_estimator_

iq_model = gscv.fit(iq_x_train, iq_y_train)
iq_best_model = gscv.best_estimator_

sj_predictions = sj_best_model.predict(sj_x_test)
iq_predictions = iq_best_model.predict(iq_x_test)

total_predictions = list(sj_predictions) + list(iq_predictions)
get_data_into_submission_format(total_predictions)

# %% Fit the pipeline

# pipe.fit(X_train_try, y_train_try)
# pipe.fit(X, y)

# # print(sqrt(mean_squared_error(pipe.predict(X_test_try), y_test_try)))


# # %% submit
# # to do fill the data
# ProcessingData.fill_data(X_test, fillType ='ffill')
# raw_prediction = pipe.predict(X_test)
# get_data_into_submission_format(raw_prediction)