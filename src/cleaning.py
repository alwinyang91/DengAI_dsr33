
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
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

# Lagging
n_lags = 25
columns = list(set(df_train_features.columns) - set(["city", "year", "weekofyear", "week_start_date", "type"]))

n_columns = len(columns)

corr_df = pd.DataFrame(index=columns, columns=list(range(1, n_lags)))

for col in columns:
    for lag in range(1, n_lags):
        new_column_name = f"lag_{lag}_{col}"
        df_train_with_label.loc[:, new_column_name] = df_train_with_label.loc[:, col].shift(lag)
        corr = df_train_with_label.loc[:, ["total_cases", new_column_name]].corr()

        corr_df.loc[col, lag] = corr.iloc[0, 1]

fig, axs = plt.subplots(figsize=(15, 15))
sns.heatmap(corr_df.astype(float), ax=axs)
fig.savefig("./corr_heatmap.png")

