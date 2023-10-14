from data_preparation import *
from model_fitting import *
import pandas as pd
import os

resources_path = '../resources'
output_path = '../output'


def pre_process(raw_data):
    return raw_data.pipe(setup) \
        .pipe(changed_order) \
        .pipe(subtract_control) \
        .pipe(subtract_start) \
        .pipe(truncate_under_threshold) \
        .pipe(add_trial_info)


def mean_each_trial(df):
    return df.groupby('trial').mean()


def save_output(preprocessed, model, file_name):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    preprocessed.to_csv('{}/{}'.format(output_path, file_name))
    model.to_csv('{}/model_{}'.format(output_path, file_name))
