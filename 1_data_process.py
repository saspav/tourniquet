import numpy as np
import pandas as pd

from pathlib import Path
from data_process_tourniquet import DataTransform, PREDICTIONS_DIR

__import__("warnings").filterwarnings('ignore')

file_dir = Path(__file__).parent
file_train = file_dir.joinpath('train.csv')
file_test = file_dir.joinpath('test.csv')
prefix_preprocess = '_MV2'

data_cls = DataTransform()
data_cls.preprocess_path_file = file_dir.joinpath(f'preprocess_df{prefix_preprocess}.pkl')

train_df = pd.read_csv(file_train, parse_dates=['timestamp'], index_col='row_id')
test_df = pd.read_csv(file_test, parse_dates=['timestamp'], index_col='row_id')
test_df.insert(0, 'user_id', -1)

all_df = pd.concat([train_df, test_df])
print(all_df.info())

# df_gt = data_cls.fit_gate_times(train_df, remake_gates_mask=True)
df_gt = data_cls.fit_gate_times(train_df, use_gates_mask_V2=True)
df_gt.to_pickle(r'D:\python-txt\tourniquet\df_gt.pkl')
df_gt.to_csv(r'D:\python-txt\tourniquet\df_gt.csv', index=False, sep=';')

# print(df_gt)

# df_gt = pd.read_pickle(r'D:\python-txt\tourniquet\df_gt.pkl')

data_cls.group_gate_times(df_gt, replace_gates_mask=True)

print('Количество паттернов:', len(data_cls.gates_mask))
print(*data_cls.gates_mask, sep='\n')

all_df = data_cls.fit(all_df, file_df=data_cls.preprocess_path_file)

all_df.to_csv(file_dir.joinpath(f'all_df{prefix_preprocess}.csv'))

all_df = pd.read_csv(file_dir.joinpath(f'all_df{prefix_preprocess}.csv'),
                     parse_dates=['timestamp'],
                     index_col='row_id')
print(all_df.columns.to_list())
