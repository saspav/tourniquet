"""Обработка файла логов: добавление скора с сайта и преобразование в эксель"""

import numpy as np
import pandas as pd
from pathlib import Path
from ast import literal_eval
from df_addons import df_to_excel

__import__("warnings").filterwarnings('ignore')

sc = pd.DataFrame(columns=['file_name', 'accuracy', 'num', 'mdl', 'local'])
file_scores = Path(r'D:\python-txt\tourniquet\site_scores.xlsx')
if file_scores.is_file():
    sc = pd.read_excel(file_scores, usecols=[2, 4])
    sc.columns = ['file_name', 'ods_acc']
    sc = sc[sc.file_name != 'predictions.csv']
    sc.file_name = sc.file_name.str.replace('CB_submit_313_local.csv', 'CB_submit_313.csv')
    sc['num'] = sc.file_name.str.findall(r'\d+').map(lambda x: int(x[0]) if x else 0)
    sc['mdl'] = sc.file_name.map(lambda x: x.split('_')[0].lower()[:2])
    sc['local'] = sc.file_name.map(lambda x: 'local' in x.lower())
    sc.sort_values(['ods_acc', 'num'], ascending=[False, True], inplace=True,
                   ignore_index=True)
    scores = sorted(sc['ods_acc'].unique(), reverse=True)
    sc['ods_n'] = sc['ods_acc'].map(lambda x: scores.index(x) + 1)

for log_postfix in ('', '_local'):
    file_logs = Path(fr'D:\python-txt\tourniquet\scores{log_postfix}.logs')

    if not file_logs.is_file():
        with open(file_logs, mode='a') as log:
            log.write('num;mdl;acc_train;acc_valid;acc_full;score;WF1;'
                      'model_columns;exclude_columns;cat_columns;comment\n')
        max_num = 0
    else:
        df = pd.read_csv(file_logs, sep=';')
        if 'acc_train' not in df.columns:
            df.insert(2, 'acc_train', 0)
            df.insert(3, 'acc_valid', 0)
            df.insert(4, 'acc_full', 0)
            df.num = df.index + 1
            df.to_csv(file_logs, index=False, sep=';')
        if 'roc_auc' not in df.columns:
            df.insert(2, 'roc_auc', 0)
            df.num = df.index + 1
            df.to_csv(file_logs, index=False, sep=';')
        df.insert(0, 'NN', 0)
        df.NN = df.index + 1

    flt_local = 'local' in log_postfix
    tmp = sc[sc['local'] == flt_local]

    df['mdl'] = df['mdl'].str.lower()
    tmp['mdl'] = tmp['mdl'].str.lower()

    df = df.merge(tmp[['mdl', 'num', 'ods_acc', 'ods_n']], on=['num', 'mdl'], how='left')
    df_columns = df.columns.to_list()[:-2]
    df_columns.insert(9, 'ods_acc')
    df_columns.insert(9, 'ods_n')
    df = df[df_columns]
    # df.fillna(0, inplace=True)
    df['ods_n'] = df['ods_n'].map(lambda x: x if x > 0 else np.NaN)

    # print(df.info())
    # print(df.isna().sum())

    ods = df[df['ods_n'] > 0][['roc_auc', 'acc_train', 'acc_valid', 'acc_full', 'score',
                               'WF1', 'ods_n', 'ods_acc']]
    # print(ods.columns)

    cols_to_be_matched = ['roc_auc', 'acc_train', 'acc_valid', 'acc_full', 'score', 'WF1']

    df = df.merge(ods, on=cols_to_be_matched, how='left')
    df['ods_n_x'] = df['ods_n_x'].fillna(df['ods_n_y'])
    df['ods_acc_x'] = df['ods_acc_x'].fillna(df['ods_acc_y'])
    df.rename(columns={'ods_n_x': 'ods_n', 'ods_acc_x': 'ods_acc'}, inplace=True)
    df.drop(['ods_n_y', 'ods_acc_y'], axis=1, inplace=True)

    fillna_columns = 'roc_auc acc_train acc_valid acc_full score WF1 ods_acc'.split()
    for col_name in fillna_columns:
        df[col_name] = df[col_name].map(lambda x: x if x > 0 else np.NaN)

    # df.sort_values(['score', 'NN'], ascending=[False, True], inplace=True)

    # экспорт в эксель
    file_xls = file_logs.with_suffix('.xlsx')
    df_to_excel(df, file_xls,
                ins_col_width=[(1, 6)] * 2 + [(3, 10)] * 6,
                float_cells=(3, 4, 5, 6, 7, 8, 10)
                )
