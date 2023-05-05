"""Исследование паттернов в турникетах"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter

from data_process_tourniquet import DataTransform

PREDICTIONS_DIR = Path(r'D:\python-txt\tourniquet\predictions')

file_dir = Path(__file__).parent
file_train = file_dir.joinpath('train.csv')
file_test = file_dir.joinpath('test.csv')

data_cls = DataTransform(category_columns=[])
data_cls.preprocess_path_file = file_dir.joinpath('preprocess_df.pkl')
# data_cls.exclude_columns = ['timestamp', 'date']

train_df = pd.read_csv(file_train, parse_dates=['timestamp'], index_col='row_id')
test_df = pd.read_csv(file_test, parse_dates=['timestamp'], index_col='row_id')
test_df.insert(0, 'user_id', -1)

all_df = pd.concat([train_df, test_df])
# print(all_df.info())

#
tmp = train_df
tmp["date"] = tmp["timestamp"].dt.date
current_user_id = current_date = prev_time = None
current_gate_times, current_gates = [], []
res_gate_times, result_gates = [], []

# Получение последовательности турникетов одного user_id в день
# смотрим только последовательности 3 и более подряд турникетов
for index, row in tqdm(tmp.iterrows()):
    if current_user_id != row["user_id"] or current_date != row["date"]:
        if len(current_gate_times) >= 3:
            res_gate_times.append((current_user_id, current_gate_times))
            result_gates.append((current_user_id, current_gates))
        current_gate_times, current_gates = [], []
        current_user_id = row["user_id"]
        current_date = row["date"]
        prev_time = row["timestamp"]
    delta = int((row["timestamp"] - prev_time).total_seconds()) if prev_time else 0
    prev_time = row["timestamp"]
    current_gate_times.append((row["gate_id"], delta))
    current_gates.append(row["gate_id"])

if len(current_gate_times) >= 3:
    res_gate_times.append((current_user_id, current_gate_times))
    result_gates.append((current_user_id, current_gates))

# print(*result_gates, sep='\n')
# print()
# print(*res_gate_times, sep='\n')

res = [*map(lambda x: tuple(x[1]), res_gate_times)]
# print(*sorted(res, key=lambda x: (tuple(p[0] for p in x),
#                                   tuple(p[1] for p in x))), sep='\n')

gates_times = [tuple(zip(*gt)) for gt in [*map(lambda x: tuple(x[1]), res_gate_times)]]
print()
print(*res[-3:], sep='\n')
print()
print(*gates_times[-3:], sep='\n')

# # возьмем последовательность меньше 7 для ручного отбора, чтобы добавить в self.gates_mask
# res_gate = [*map(lambda x: tuple(x[1][:6]), result_gates)]
# получение из последовательность турникетов шаблонов длиной от 3х до 6-ти турникетов

use_thresholds = False  # использовать ограничения по кол-ву последовательных турникетов

res_gate = []
for user_gates in result_gates:
    gates = user_gates[1]
    if use_thresholds:
        start_range = 3 if len(gates) < 5 else 4
    else:
        start_range = 3
    for len_mask in range(start_range, 7):
        res_gate.extend([*zip(*[gates[i:] for i in range(len_mask)])])
print()
print(*res_gate[-3:], sep='\n')
print('Количество комбинаций в шаблонах:', len(res_gate))
# print()
# print(*sorted(res_gate)[-3:], sep='\n')
# print()

dfg = pd.DataFrame(columns='gates;len_g;cnt_g;dbl'.split(';'))

res_cnt = Counter(res_gate)
with open(r'D:\python-txt\tourniquet\gates.csv', mode='w') as file:
    file.write('gates;len_g;cnt_g;dbl\n')
    prev_key = prev_cnt = None
    find_gates_mask = []
    # записываем минимальную длину шаблонов при сдвиге на 1 вправо
    for key, cnt in sorted(res_cnt.items()):
        # print(key, '-->', len(key), '-->', cnt)
        if use_thresholds:
            if (cnt in (3, 4) and len(key) in (3, 4)) or (cnt == 5 and len(key) < 6) or (
                    cnt > 5 and len(key) < cnt):
                doubles = 0
                if len(key) > 4 and prev_key == key[:-1] and abs(prev_cnt - cnt) < 3:
                    print(prev_key, key, cnt)
                    prev_key, prev_cnt = key, cnt
                    doubles = 1
                    continue
                prev_key, prev_cnt = key, cnt
                find_gates_mask.append(key)
                file.write(f'{key};{len(key)};{cnt};{doubles}\n')
        else:
            doubles = 0
            if len(key) > 4 and prev_key == key[:-1] and abs(prev_cnt - cnt) < 3:
                prev_key, prev_cnt = key, cnt
                doubles = 1
            prev_key, prev_cnt = key, cnt
            find_gates_mask.append(key)
            file.write(f'{key};{len(key)};{cnt};{doubles}\n')

        dfg.loc[len(dfg)] = [key, len(key), cnt, doubles]

print('Уникальных комбинаций:', len(res_cnt))

# print(Counter(res_cnt.values()))

# прочитаем существующие колонки с масками
prefix_preprocess = '_min_0'
all_df = pd.read_csv(file_dir.joinpath(f'all_df{prefix_preprocess}.csv'),
                     parse_dates=['timestamp'],
                     index_col='row_id')
gate_cols = [col for col in all_df.columns.to_list() if col.startswith('G')]
gate_cols = [*map(lambda x: tuple(map(int, x[1:].split('_'))), gate_cols)]
dfm = pd.DataFrame(data=zip(gate_cols, [1] * len(gate_cols)), columns=['gates', 'cls'])

dfg = dfg.merge(dfm, on='gates', how='left')
dfg.fillna(0, inplace=True)
dfg['cls'] = dfg['cls'].astype(int)
dfg.to_csv(r'D:\python-txt\tourniquet\dfg.csv', sep=';', index=False)
