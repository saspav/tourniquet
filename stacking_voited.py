import os
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from data_process_tourniquet import DataTransform, PREDICTIONS_DIR
from data_process_tourniquet import predict_train_valid, predict_test, get_max_num
from print_time import print_time, print_msg

__import__("warnings").filterwarnings('ignore')


def merge_files(name_files, data_train=False):
    df = pd.DataFrame()
    for filename in name_files:
        print('Обрабатываю файл:', Path(filename).name)
        if data_train:
            temp = pd.read_csv(filename)
            temp.drop('row_id', axis=1, inplace=True)
        else:
            temp = pd.read_csv(filename, index_col='row_id')
        col_prefix = Path(filename).name[:2]
        target_col = f'{col_prefix}_target'
        temp.rename(columns={'target': target_col}, inplace=True)
        if len(df):
            df = df.merge(temp[[target_col]], how='left', left_index=True, right_index=True)
        else:
            temp_columns = ['user_id', target_col] if data_train else [target_col]
            df = temp[temp_columns]

    print('df.shape:', df.shape)

    df['voited'] = df.mode(axis=1)[0].astype(int)
    return df


def save_submit(df, column, max_num=0):
    submit_prefix = 'voited' if column == 'voited' else 'mux'
    submit_csv = f'{submit_prefix}_submit_{max_num:03}_local.csv'
    file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)

    # Write the predictions to a file
    submit = df[[column]]
    submit['target'] = df[[column]]
    submit[['target']].to_csv(file_submit_csv)
    roc_auc = acc_train = acc_valid = score = f1w = 0
    acc_full = accuracy_score(train_df['user_id'], train_df[column])
    with open(file_logs, mode='a') as log:
        # log.write('num;mdl;roc_auc;acc_train;acc_valid;acc_full;score;WF1;'
        #           'model_columns;exclude_columns;cat_columns;comment\n')
        log.write(f'{max_num};{submit_prefix[:2]};{roc_auc:.6f};'
                  f'{acc_train:.6f};{acc_valid:.6f};{acc_full:.6f};'
                  f'{score:.6f};{f1w:.6f};{train_df.columns.tolist()};'
                  f'{data_cls.exclude_columns};{cat_columns};{data_cls.comment}\n')


file_logs = Path(r'D:\python-txt\tourniquet\scores_local.logs')
max_num = get_max_num(file_logs) + 1

start_time = print_msg('Голосование по большинству голосов среди классификаторов...')

stacking_dir = PREDICTIONS_DIR.joinpath('voited')
files_train = glob(f'{stacking_dir}/*train*.csv')
files_test = glob(f'{stacking_dir}/*submit*.csv')

train_df = merge_files(files_train, data_train=True)
test_df = merge_files(files_test)
model_columns = test_df.columns.to_list()

print(f'Размер test_df = {test_df.shape}')

cat_columns = []
data_cls = DataTransform(use_catboost=True, category_columns=cat_columns)
data_cls.exclude_columns = []
data_cls.comment = {'files': files_test}
save_submit(test_df, 'voited', max_num)

acc_voited = accuracy_score(train_df['user_id'], train_df['voited'])

print(f'Accuracy voited:{acc_voited}')
