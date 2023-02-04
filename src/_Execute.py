import numpy as np

# import cleaned data
# df_x = ...
# df_y = ...


# training models
split_size = 0.7
X_train = df_x.iloc[np.int(split_size*len(df_x)):, :]
y_train = df_y.iloc[np.int(split_size*len(df_x)):, :]
X_val = df_x.iloc[:np.int(split_size*len(df_x)), :]
y_val = df_y.iloc[:np.int(split_size*len(df_x)), :]


# training
# evaluating models

# pick the best model to predict y_test
# training again
# predict y_test


# create submission file

subm = pd.read_csv('./submission_format.csv')

labels = pd.DataFrame({'total_cases': pd.Series(lr_predict)})
subm.loc[:, 'total_cases'] = labels.astype(int)

#subm.reset_index(index=False)
#export to csv file

subm.to_csv('submission_230202.csv', index=False)
subm