train_features_file = './data/dengue_features_test.csv'
train_label_file = './data/dengue_labels_train.csv'
test_features_file = './dengue_features_test.csv'

label_columns = 'total_cases'
numeric_columns = ['year',
                   'weekofyear',
                   'ndvi_ne',
                   'ndvi_nw',
                   'ndvi_se',
                   'ndvi_sw',
                   'precipitation_amt_mm',
                   'reanalysis_air_temp_k',
                   'reanalysis_avg_temp_k',
                   'reanalysis_dew_point_temp_k',
                   'reanalysis_max_air_temp_k',
                   'reanalysis_min_air_temp_k',
                   'reanalysis_precip_amt_kg_per_m2',
                   'reanalysis_relative_humidity_percent',
                   'reanalysis_sat_precip_amt_mm',
                   'reanalysis_specific_humidity_g_per_kg',
                   'reanalysis_tdtr_k',
                   'station_avg_temp_c',
                   'station_diur_temp_rng_c',
                   'station_max_temp_c',
                   'station_min_temp_c',
                   'station_precip_mm']
categorical_columns  = ['city']
all_columns = [label_columns] + categorical_columns + numeric_columns
all_columns_nolabel = categorical_columns + numeric_columns
categorical = {
    'city': ['sj', 'iq']
}

cols_to_norm = ['precipitation_amt_mm',
                'reanalysis_air_temp_k',
                'reanalysis_avg_temp_k',
                'reanalysis_dew_point_temp_k',
                'reanalysis_max_air_temp_k',
                'reanalysis_min_air_temp_k',
                'reanalysis_precip_amt_kg_per_m2',
                'reanalysis_relative_humidity_percent',
                'reanalysis_sat_precip_amt_mm',
                'reanalysis_specific_humidity_g_per_kg',
                'reanalysis_tdtr_k',
                'station_avg_temp_c',
                'station_diur_temp_rng_c',
                'station_max_temp_c',
                'station_min_temp_c',
                'station_precip_mm']
cols_to_scale = ['year',
                 'weekofyear']

date_columns = "week_start_date"
train_features_file = './data/dengue_features_test.csv'
train_label_file = './data/dengue_labels_train.csv'
test_features_file = './dengue_features_test.csv'
