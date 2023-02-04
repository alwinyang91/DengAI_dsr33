import pandas as pd

class LoadData:

    def __init__(self, input_data_path1,input_data_path2,input_data_path3):
        self.input_data_path1 = input_data_path1
        self.input_data_path2 = input_data_path2
        self.input_data_path3 = input_data_path3

    def read_dataframe(self, joinlabe, jointest):

        df_features = pd.read_csv(self.input_data_path1)
        df_label = pd.read_csv(self.input_data_path2)
        df_test = pd.read_csv(self.input_data_path3)

        if joinlabe:
            df_train = pd.concat([df_features, df_label.loc[:,'total_cases']], axis=1)
        else:
            df_train = df_features

        if jointest:
            df_train = pd.concat([df_train, df_test], axis=0)
        else:
            pass

        return df_train


    # df_features = pd.read_csv('./data/dengue_features_train.csv',)
    # print('Features data is loaded, named: df_features.')
    # # print('Infromaion about the df\n', df_features.info(verbose=True, show_counts=True))
    
    # df_label = pd.read_csv('./data/dengue_labels_train.csv',)
    # print('Label data is loaded, named:\n df_label.')
    # # print('Infromaion about the df\n', df_label.info(verbose=True, show_counts=True))

    # df_train = pd.concat([df_features, df_label.loc[:,'total_cases']], axis=1)
    # print('Concat the df_features and df_label, named:\n df_train.')

    # df_test = pd.read_csv('./data/dengue_features_test.csv',)
    # print('Test data is loaded, named:\n df_test.')
    # # print('Infromaion about the df\n', df_test.info(verbose=True, show_counts=True))

