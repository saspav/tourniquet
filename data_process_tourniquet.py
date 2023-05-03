import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, datetime, timedelta
from tqdm import tqdm
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from df_addons import memory_compression
from print_time import print_time, print_msg

__import__("warnings").filterwarnings('ignore')

PREDICTIONS_DIR = Path(r'D:\python-txt\tourniquet\predictions')


def get_max_num(file_logs=None):
    if file_logs is None:
        file_logs = Path(r'D:\python-txt\tourniquet\scores_local.logs')

    if not file_logs.is_file():
        with open(file_logs, mode='a') as log:
            log.write('num;mdl;roc_auc;acc_train;acc_valid;acc_full;score;WF1;'
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
        df.num = df.index + 1
        max_num = df.num.max()
    return max_num


def predict_train_valid(model, datasets, label_enc=None):
    X_train, X_valid, y_train, y_valid, train, target, test_df = datasets
    valid_pred = model.predict(X_valid)
    train_pred = model.predict(X_train)
    train_full = model.predict(train)

    if len(valid_pred.shape) > 1 and valid_pred.shape[1] > 1:
        predict_proba = valid_pred.copy()
        valid_pred = np.argmax(valid_pred, axis=1)
    else:
        predict_proba = model.predict_proba(X_valid)

    if len(train_pred.shape) > 1 and train_pred.shape[1] > 1:
        train_pred = np.argmax(train_pred, axis=1)
    if len(train_full.shape) > 1 and train_full.shape[1] > 1:
        train_full = np.argmax(train_full, axis=1)

    f1w = f1_score(y_valid, valid_pred, average='weighted')
    acc_valid = accuracy_score(y_valid, valid_pred)
    acc_train = accuracy_score(y_train, train_pred)
    acc_full = accuracy_score(target, train_full)
    try:
        roc_auc = roc_auc_score(y_valid, predict_proba,
                                average='weighted', multi_class='ovr')
    except:
        roc_auc = 0

    # print(classification_report(y_valid, valid_pred))
    return acc_train, acc_valid, acc_full, roc_auc, f1w


def predict_test(idx_fold, model, datasets, max_num=0, submit_prefix='lg_', label_enc=None,
                 save_predict_proba=True):
    X_train, X_valid, y_train, y_valid, train, target, test_df = datasets
    # постфикс если было обучение на отдельных фолдах
    nfld = f'_{idx_fold}' if idx_fold else ''
    predictions = model.predict(test_df)
    predict_train = model.predict(train)

    if label_enc:
        predictions = label_enc.inverse_transform(predictions)
        predict_train = label_enc.inverse_transform(predict_train)

    # печать размерности предсказаний и списка меток классов
    classes = model.classes_.tolist()
    print('predict_proba.shape:', predictions.shape, 'classes:', classes)

    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predict_proba = predictions.copy()
        predictions = np.argmax(predictions, axis=1)
        train_proba = predict_train.copy()
        predict_train = np.argmax(predict_train, axis=1)
    else:
        predict_proba = model.predict_proba(test_df)
        train_proba = model.predict_proba(train)

    submit_csv = f'{submit_prefix}submit_{max_num:03}{nfld}_local.csv'
    file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)
    file_proba_csv = PREDICTIONS_DIR.joinpath(submit_csv.replace('submit_', 'proba_'))
    file_train_csv = PREDICTIONS_DIR.joinpath(submit_csv.replace('submit_', 'train_'))

    # Write the predictions to a file
    submit = test_df[test_df.columns.to_list()[5:7]]
    submit['target'] = predictions
    submit[['target']].to_csv(file_submit_csv)
    if save_predict_proba:
        train_sp = pd.DataFrame(target)
        train_sp['target'] = predict_train
        train_sp.to_csv(file_train_csv)

        try:
            train_sp[classes] = train_proba
            train_sp.to_csv(file_train_csv)
        except:
            pass

        try:
            proba = submit[['target']]
            proba[classes] = predict_proba
            proba.to_csv(file_proba_csv)
        except:
            pass

    acc_train, acc_valid, acc_full, roc_auc, f1w = predict_train_valid(model, datasets,
                                                                       label_enc=label_enc)

    print(f'Accuracy = {acc_valid:.6f}')
    print(f'Weighted F1-score = {f1w:.6f}')

    print(f'Accuracy train:{acc_train} valid:{acc_valid} full:{acc_full} roc_auc:{roc_auc}')
    return acc_train, acc_valid, acc_full, roc_auc, f1w


class DataTransform:
    def __init__(self, use_catboost=False, numeric_columns=None, category_columns=None,
                 drop_first=False, scaler=None, args_scaler=None):
        """
        Преобразование данных
        :param use_catboost: данные готовятся для catboost
        :param numeric_columns: цифровые колонки
        :param category_columns: категориальные колонки
        :param drop_first: из dummy переменных удалить первую колонку
        :param scaler: какой скайлер будем использовать
        :param degree: аргументы для скайлера,
        например: степень для полиномиального преобразования
        """
        self.use_catboost = use_catboost
        self.category_columns = [] if category_columns is None else category_columns
        self.numeric_columns = [] if numeric_columns is None else numeric_columns
        self.drop_first = drop_first
        self.exclude_columns = []
        self.new_columns = []
        self.comment = {'drop_first': drop_first}
        self.train_months = (7, 8, 9, 10,)
        self.valid_months = (11, 12)
        self.train_idxs = None
        self.valid_idxs = None
        self.transform_columns = None
        self.scaler = scaler
        self.args_scaler = args_scaler
        self.preprocess_path_file = None
        self.beep_outlet = None
        self.gates_mask = [(-1, -1, -1), (-1, -1, 10), (-1, -1, 11), (3, 3, 4), (3, 3, 10),
                           (3, 3, 10, 11), (4, 4, 3), (4, 4, 4), (4, 4, 5), (4, 4, 7),
                           (4, 4, 8), (4, 4, 9, 9), (4, 4, 9, 9, 5, 5), (4, 7, 3), (4, 9, 9),
                           (5, 5, 10), (5, 10, 11), (6, 3, 3), (6, 6, 5), (6, 6, 7),
                           (6, 6, 9, 9), (6, 7, 3), (6, 9, 9), (7, 3, 3), (7, 3, 3, 10),
                           (7, 3, 3, 10, 11), (7, 3, 3, 11), (7, 3, 10), (7, 5, 5),
                           (7, 5, 5, 10), (7, 5, 5, 10, 11), (8, 8, 5), (7, 8, 8), (7, 9, 9),
                           (7, 9, 9, 3, 3), (7, 9, 9, 5, 5), (7, 9, 9, 5, 5, 5),
                           (7, 9, 9, 5, 5, 10), (9, 5, 5), (9, 5, 5, 10), (9, 9, 3),
                           (9, 9, 5), (9, 9, 5, 5), (9, 9, 5, 5, 10), (9, 9, 15),
                           (10, 11, 4, 4), (10, 11, 6, 6), (10, 11, 10), (10, 13, 13),
                           (11, 4, 4), (11, 4, 4, 3), (11, 4, 4, 3, 3), (11, 4, 4, 3, 3, 10),
                           (11, 4, 4, 4), (11, 4, 4, 4, 4), (11, 4, 4, 5), (11, 4, 4, 5, 5),
                           (11, 4, 4, 5, 5, 10), (11, 4, 4, 7), (11, 4, 4, 7, 3, 3),
                           (11, 4, 4, 7, 5), (11, 4, 4, 8, 8), (11, 4, 4, 9, 9),
                           (11, 4, 4, 9, 9, 5), (11, 4, 4, 9, 9, 15), (11, 4, 4, 15),
                           (11, 6, 6), (11, 6, 6, 5), (11, 6, 6, 6), (11, 6, 6, 9, 9),
                           (11, 10, 11), (11, 10, 11, 4), (11, 11, 4, 4), (11, 11, 4, 4, 9),
                           (11, 11, 10), (12, 12, 11), (12, 12, 11, 4), (13, 13, 4, 4),
                           (13, 13, 6, 6), (13, 13, 10), (13, 13, 11), (13, 13, 12, 12),
                           (13, 13, 12, 12, 11), (15, 3, 3), (15, 3, 3, 10),
                           (15, 3, 3, 10, 11), (15, 9, 9), (15, 9, 9, 5, 5),
                           ]
        self.gates_M_V2 = [(-1, -1, -1), (-1, -1, -1, -1), (-1, -1, 10), (-1, -1, 11),
                           (3, 3, 4), (3, 3, 10), (3, 3, 10, 11), (3, 3, 10, 11, 4),
                           (3, 3, 10, 11, 6), (3, 4, 4), (3, 10, 11), (3, 10, 11, 4, 4),
                           (3, 10, 11, 6), (4, 3, 3), (4, 3, 3, 10), (4, 4, 3), (4, 4, 4),
                           (4, 4, 3, 3, 10), (4, 4, 4, 9), (4, 4, 5), (4, 4, 5, 5, 10),
                           (4, 4, 7), (4, 4, 7, 3, 3), (4, 4, 7, 5), (4, 4, 8), (4, 4, 9, 9),
                           (4, 4, 9, 9, 5, 5), (4, 4, 9, 9, 15), (4, 4, 15), (4, 5, 5),
                           (4, 5, 5, 10), (4, 7, 3), (4, 7, 3, 3, 10), (4, 7, 5), (4, 8, 8),
                           (4, 9, 9), (4, 9, 9, 5, 5), (4, 9, 9, 5, 5, 10), (4, 9, 9, 15),
                           (5, 5, 5), (5, 5, 5, 10), (5, 5, 10), (5, 5, 10, 11), (5, 10, 11),
                           (5, 5, 10, 11, 4, 4), (5, 5, 10, 11, 6), (5, 5, 10, 13),
                           (5, 10, 11, 4, 4), (5, 10, 11, 4, 4, 9), (5, 10, 11, 6, 6),
                           (5, 10, 13), (6, 3, 3), (6, 5, 5), (6, 6, 5), (6, 6, 6),
                           (6, 6, 6, 6, 9), (6, 6, 7), (6, 6, 9, 9), (6, 7, 3), (6, 9, 9),
                           (6, 9, 9, 5), (7, 3, 3), (7, 3, 3, 10), (7, 3, 3, 10, 11),
                           (7, 3, 3, 10, 11, 4), (7, 3, 10), (7, 5, 5), (7, 5, 5, 10),
                           (7, 5, 5, 10, 11), (7, 8, 8), (7, 9, 9), (7, 9, 9, 3, 3),
                           (7, 9, 9, 5, 5), (7, 9, 9, 5, 5, 5), (7, 9, 9, 5, 5, 10),
                           (8, 8, 3), (8, 8, 5), (9, 3, 3), (9, 5, 5), (9, 5, 5, 5, 5),
                           (9, 5, 5, 10), (9, 5, 5, 10, 11), (9, 5, 5, 10, 11, 4), (9, 9, 3),
                           (9, 5, 5, 10, 11, 6), (9, 5, 5, 10, 13), (9, 9, 5), (9, 9, 5, 5),
                           (9, 9, 5, 5, 5, 5), (9, 9, 5, 5, 10), (9, 9, 15), (10, 11, 4, 4),
                           (10, 11, 4, 4, 9, 9), (10, 11, 6, 6), (10, 11, 10), (10, 13, 13),
                           (11, 4, 4), (11, 4, 4, 3), (11, 4, 4, 3, 3), (11, 4, 4, 3, 3, 10),
                           (11, 4, 4, 4), (11, 4, 4, 4, 4), (11, 4, 4, 5), (11, 4, 4, 5, 5),
                           (11, 4, 4, 5, 5, 10), (11, 4, 4, 7), (11, 4, 4, 7, 3, 3),
                           (11, 4, 4, 7, 5), (11, 4, 4, 8, 8), (11, 4, 4, 9, 9), (11, 6, 6),
                           (11, 4, 4, 9, 9, 5), (11, 4, 4, 9, 9, 15), (11, 4, 4, 15),
                           (11, 6, 6, 5), (11, 6, 6, 6), (11, 6, 6, 9, 9), (11, 10, 11),
                           (11, 10, 11, 4), (11, 11, 4, 4), (11, 11, 4, 4, 9), (12, 12, 11),
                           (12, 11, 4, 4), (12, 12, 11, 4), (13, 4, 4), (13, 12, 12),
                           (13, 13, 4, 4), (13, 13, 6, 6), (13, 13, 10), (13, 13, 11),
                           (13, 13, 12, 12), (13, 13, 12, 12, 11), (15, 3, 3),
                           (15, 3, 3, 10), (15, 3, 3, 10, 11), (15, 9, 9), (15, 9, 9, 5, 5),
                           ]

    def initial_preparation(self, df):
        """
        Общая первоначальная подготовка данных
        :param df: исходный ДФ
        :return: обработанный ДФ
        """
        df["date"] = df["timestamp"].dt.date
        df["time"] = df["timestamp"].dt.time
        df["day"] = df["timestamp"].dt.day
        df["week"] = df["timestamp"].dt.week
        df["month"] = df["timestamp"].dt.month

        df["hour"] = df["timestamp"].dt.hour
        df["min"] = df["timestamp"].dt.minute
        df["sec"] = df["timestamp"].dt.second

        df['minutes'] = df["hour"] * 60 + df["min"]
        df['seconds'] = df.minutes * 60 + df["sec"]

        # 1-й день месяца
        df["1day"] = df["timestamp"].dt.is_month_start.astype(int)
        # 2-й день месяца
        df["2day"] = (df.day == 2).astype(int)
        # предпоследний день месяца
        df["last_day-1"] = (df.day == df.timestamp.dt.daysinmonth - 1).astype(int)
        # Последний день месяца
        df["last_day"] = df["timestamp"].dt.is_month_end.astype(int)

        df["weekday"] = df["timestamp"].dt.dayofweek  # День недели от 0 до 6
        df["dayofweek"] = df["weekday"] + 1  # День недели от 1 до 7

        # Метка выходного дня
        df["is_weekend"] = df["weekday"].map(lambda x: 1 if x in (5, 6) else 0)

        # метки "график 2 через 2"
        df["DofY1"] = (df["timestamp"].dt.dayofyear % 4).apply(lambda x: int(x in (1, 2)))
        df["DofY2"] = (df["timestamp"].dt.dayofyear % 4).apply(lambda x: int(x < 2))

        # df['morning'] = df['hour'].map(lambda x: 1 if 6 <= x <= 10 else 0)
        # df['daytime'] = df['hour'].map(lambda x: 1 if 11 <= x <= 17 else 0)
        # df['evening'] = df['hour'].map(lambda x: 1 if 18 <= x <= 22 else 0)
        # df['night'] = df['hour'].map(lambda x: 1 if 0 <= x <= 5 or x == 23 else 0)

        # Подсчет количества срабатываний за день
        df["beep_count"] = df.groupby("date").timestamp.transform("count")
        # Подсчет количества срабатываний за день по каждому gate_id
        df["beep_gate"] = df.groupby(["date", "gate_id"]).timestamp.transform("count")

        return df

    def cat_dummies(self, df):
        """
        Отметка категориальных колонок --> str для catboost
        OneHotEncoder для остальных
        :param df: ДФ
        :return: ДФ с фичами
        """
        # если нет цифровых колонок --> заполним их
        if self.category_columns and not self.numeric_columns:
            self.numeric_columns = [col_name for col_name in df.columns
                                    if col_name not in self.category_columns]
        # если нет категориальных колонок --> заполним их
        if self.numeric_columns and not self.category_columns:
            self.category_columns = [col_name for col_name in df.columns
                                     if col_name not in self.numeric_columns]

        for col_name in self.category_columns:
            if col_name in df.columns:
                if self.use_catboost:
                    df[col_name] = df[col_name].astype(str)
                else:
                    print(f'Трансформирую колонку: {col_name}')
                    # Create dummy variables
                    df = pd.get_dummies(df, columns=[col_name], drop_first=self.drop_first)

                    self.new_columns.extend([col for col in df.columns
                                             if col.startswith(col_name)])
        return df

    def apply_scaler(self, df):
        """
        Масштабирование цифровых колонок
        :param df: исходный ДФ
        :return: нормализованный ДФ
        """
        if not self.transform_columns:
            self.transform_columns = self.numeric_columns
        if self.scaler and self.transform_columns:
            print(f'Применяю scaler: {self.scaler.__name__} '
                  f'с аргументами: {self.args_scaler}')
            args = self.args_scaler if self.args_scaler else tuple()
            scaler = self.scaler(*args)
            scaled_data = scaler.fit_transform(df[self.transform_columns])
            if scaled_data.shape[1] != len(self.transform_columns):
                print(f'scaler породил: {scaled_data.shape[1]} колонок')
                new_columns = [f'pnf_{n:02}' for n in range(scaled_data.shape[1])]
                df = pd.concat([df, pd.DataFrame(scaled_data, columns=new_columns)], axis=1)
                self.exclude_columns.extend(self.transform_columns)
            else:
                df[self.transform_columns] = scaled_data

            self.comment.update(scaler=self.scaler.__name__, args_scaler=self.args_scaler)
        return df

    def fit_gate_times(self, df, remake_gates_mask=False, use_gates_mask_V2=False):
        """
        Получение паттернов прохода через турникеты
        :param df: тренировочный ДФ
        :param remake_gates_mask: получить шаблоны масок из трейна, иначе взять из класса
        :param use_gates_mask_V2: использовать расширенный набор масок из класса
        :return: ДФ с паттернами
        """
        print('Ищу паттерны в данных...\n')
        current_user_id = prev_time = None
        current_gate_times, current_gates = [], []
        res_gate_times, result_gates = [], []

        for _, row in tqdm(df.iterrows()):
            if current_user_id != row["user_id"]:
                if len(current_gate_times) >= 3:
                    res_gate_times.append((current_user_id, current_gate_times))
                    result_gates.append((current_user_id, current_gates))
                current_gate_times, current_gates = [], []
                current_user_id = row["user_id"]
                prev_time = row["timestamp"]
            delta = int((row["timestamp"] - prev_time).total_seconds()) if prev_time else 0
            prev_time = row["timestamp"]
            current_gate_times.append((row["gate_id"], delta))
            current_gates.append(row["gate_id"])

        if len(current_gate_times) >= 3:
            res_gate_times.append((current_user_id, current_gate_times))
            result_gates.append((current_user_id, current_gates))

        gates_times = [tuple(zip(*gt)) for gt in
                       [*map(lambda x: tuple(x[1]), res_gate_times)]]

        if remake_gates_mask:
            res_gate = []
            for user_gates in result_gates:
                gates = user_gates[1]
                start_range = 3 if len(gates) < 5 else 4
                for len_mask in range(start_range, 7):
                    res_gate.extend([*zip(*[gates[i:] for i in range(len_mask)])])
            res_cnt = Counter(res_gate)
            prev_key = prev_cnt = None
            find_gates_mask = []
            for key, cnt in sorted(res_cnt.items()):
                # количество шаблонов 1 или 2 - игнорируем,
                # количество шаблонов 3 и 4 берем только длиной 3 и 4,
                # количество шаблонов 5 берем длины 3, 4 и 5,
                # количество шаблонов 6 и более берем все
                if (cnt in (3, 4) and len(key) in (3, 4)) or (cnt == 5 and len(key) < 6) or (
                        cnt > 5 and len(key) < cnt):
                    # убираем дубли когда след шаблон отличается на последним турникетом
                    # и количество шаблонов различается на 2 и менее
                    if len(key) > 4 and prev_key == key[:-1] and abs(prev_cnt - cnt) < 3:
                        prev_key, prev_cnt = key, cnt
                        continue
                    prev_key, prev_cnt = key, cnt
                    find_gates_mask.append(key)
            # заменим ручной отбор шаблонов на автоматический
            self.gates_mask = find_gates_mask
            # print(*self.gates_mask, sep='\n')
        if use_gates_mask_V2:
            # заменим ручной отбор шаблонов на ручной отбор V2
            self.gates_mask = self.gates_M_V2
            # print(*self.gates_mask, sep='\n')
        print('Количество паттернов:', len(self.gates_mask))

        df_gt = pd.DataFrame(columns=['mask'] + [f'dt_{i}' for i in range(6)])
        for gates, times in tqdm(gates_times):
            for mask in self.gates_mask:
                for idx, sub_gates in enumerate(zip(*[gates[i:] for i in range(len(mask))])):
                    if sub_gates == mask:
                        row_df_gt = [mask] + [*times[idx:idx + len(mask)]] + [0] * 3
                        df_gt.loc[len(df_gt)] = row_df_gt[:7]
        df_gt['dt_0'] = 0
        return df_gt

    def group_gate_times(self, df_gt, replace_gates_mask=False):
        """
        Получение временных интервалов для паттернов
        :param df_gt: ДФ с паттернами
        :param replace_gates_mask: заменить атрибут self.gates_mask
        :return: список паттернов с временными интервалами
        """

        # диапазон границ интервалов расширим вниз на 50% и вверх 20% - это сработало лучше,
        # чем расширение границ вниз и вверх на 5%
        def make_min_max(col):
            min_col = min(col)
            min_col = 0 if min_col < 10 else int(min_col * 0.5)  # добавил вот это
            return min_col, int(max(col) * 1.2)

        grp = df_gt.groupby('mask', as_index=False).agg(
            min_max_0=('dt_0', lambda x: make_min_max(x)),
            min_max_1=('dt_1', lambda x: make_min_max(x)),
            min_max_2=('dt_2', lambda x: make_min_max(x)),
            min_max_3=('dt_3', lambda x: make_min_max(x)),
            min_max_4=('dt_4', lambda x: make_min_max(x)),
            min_max_5=('dt_5', lambda x: make_min_max(x)),
        )
        result = []
        grp_columns = grp.columns.to_list()
        for _, row in grp.iterrows():
            mask = row['mask']
            result.append((mask, tuple(row[col] for col in grp_columns[1:len(mask) + 1])))

        if replace_gates_mask:
            self.gates_mask = result

        return result

    @staticmethod
    def find_gates(row, mask, times=None):
        shift_gates = [f'g{i}' for i in range(len(mask) - 1, -len(mask), -1)]
        gates = row[shift_gates].values
        gates_times = None
        if times:
            shift_times = [f't{i}' for i in range(len(mask) - 1, -len(mask), -1)]
            gates_times = row[shift_times].values
        index_mask = -1
        for idx, sub_gates in enumerate(zip(*[gates[i:] for i in range(len(mask))])):
            # найден паттерн турникетов
            if sub_gates == mask:
                # если есть проверка по дельта времени прохода --> смотрим,
                # чтобы дельта попадала в границы диапазона обученных паттернов
                if times is None or (
                        times and all(times[i][0] <= gates_times[idx + i] <= times[i][1]
                                      for i in range(1, len(mask)))):
                    # print('паттерн найден')
                    index_mask = idx
                    break
                print('паттерн не найден, user_id:', row['user_id'], 'idx =', idx)
                print('mask, gates:', mask, gates)
                print('times, gates_times:', times, gates_times)
        return index_mask >= 0

    def fit(self, df, file_df=None, out_five_percent=False, remake_gates_mask=False):
        """
        Формирование фич
        :param df: исходный ФД
        :param file_df: Предобработанный Файл .pkl с полным путём
        :param out_five_percent: граница 5% при определении выбросов
        :param remake_gates_mask: получить шаблоны масок из трейна, иначе взять из класса
        :return: обработанный ДФ
        """
        if file_df and file_df.suffix == '.pkl' and file_df.is_file():
            df = pd.read_pickle(file_df)
            return df

        df = self.initial_preparation(df)

        # данные для устранения выбросов, где рабочий день помечен как выходной и наоборот
        tmp = df[['date', 'weekday', 'beep_count', 'is_weekend']].drop_duplicates()
        tmp["weekend"] = tmp["weekday"].map(lambda x: 1 if x in (5, 6) else 0)
        beep_cnt = tmp[tmp["weekend"] == 1].beep_count
        if out_five_percent:
            self.beep_outlet = beep_cnt.quantile(0.975)  # 69.75
        else:
            self.beep_outlet = beep_cnt.quantile(0.75) + beep_cnt.std() * 1.5  # 98.7

        # выделил shift по датам, чтобы случайно не зацепить переход между сутками
        result = pd.DataFrame()
        for flt_date in sorted(df["date"].unique()):
            tmp = df[df["date"] == flt_date]
            # формирование колонок с gate_id для 5 предыдущих и следующих строк
            for i in range(5, -6, -1):
                tmp[f'g{i}'] = tmp['gate_id'].shift(i, fill_value=-9)

            # формирование колонок с timestamp для 5 предыдущих и следующих строк
            for i in range(5, -6, -1):
                tmp[f'ts{i}'] = tmp['timestamp'].shift(i)
            tmp[f'ts6'] = tmp[f'ts5']
            for i in range(5, -6, -1):
                tmp[f't{i}'] = tmp[f'ts{i}'] - tmp[f'ts{i + 1}']
                tmp[f't{i}'] = tmp[f't{i}'].map(lambda x: x.total_seconds())
                tmp[f't{i}'].fillna(0, inplace=True)
                tmp[f't{i}'] = tmp[f't{i}'].astype(int)

            # удалить временные колонки с timestamp.shift()
            tmp.drop([f'ts{i}' for i in range(6, -6, -1)], axis=1, inplace=True)

            if not len(result):
                result = tmp
            else:
                result = pd.concat([result, tmp])

        df = result

        # df.to_csv('df_ts.csv', sep=';')

        if remake_gates_mask:
            # получим паттерны
            tmp_columns = ['user_id', 'timestamp', 'gate_id']
            train_tmp = df[df.user_id > -1][tmp_columns]
            self.fit_gate_times(train_tmp, remake_gates_mask=True)
            print('Количество паттернов:', len(self.gates_mask))

        start_time = print_msg('Поиск по шаблонам...')

        tqdm.pandas()
        for mask in self.gates_mask:
            times = None
            if len(mask) == 2:
                mask, times = mask
            mask_col = 'G' + '_'.join(map(str, mask))
            print(f'Шаблон: {mask} колонка: {mask_col}')
            df[mask_col] = df.progress_apply(
                lambda row: self.find_gates(row, mask, times=times), axis=1).astype(int)

        if self.preprocess_path_file:
            df.to_pickle(self.preprocess_path_file)
            df.to_csv(self.preprocess_path_file.with_suffix('.csv'))

        print_time(start_time)
        return df

    def transform(self, df, model_columns=None):
        """
        Формирование фич
        :param df: ДФ
        :param model_columns: список колонок, которые будут использованы в модели
        :return: ДФ с фичами
        """
        df = self.initial_preparation(df)

        # устранение выбросов, где рабочий день помечен как выходной и наоборот
        if self.beep_outlet:
            weekend_to_work = df["is_weekend"].eq(1) & df["beep_count"].gt(self.beep_outlet)
            work_to_weekend = df["is_weekend"].eq(0) & df["beep_count"].lt(self.beep_outlet)
            df.loc[weekend_to_work, 'is_weekend'] = 0
            df.loc[work_to_weekend, 'is_weekend'] = 1

        # выделение временных лагов между проходами через gate_id
        lags = {'lag0': lambda x: not x, 'lag1': lambda x: x == 1, 'lag2': lambda x: x == 2,
                'lag3': lambda x: x <= 3,
                'lag4': lambda x: 2 < x <= 5,
                'lag5': lambda x: 5 < x <= 15,
                'lag6': lambda x: 15 < x <= 25,
                'lag7': lambda x: 25 < x <= 36,
                'lag8': lambda x: 36 < x <= 49,
                'lag9': lambda x: 49 < x <= 79,
                }

        for col_name, lag_func in lags.items():
            df[col_name] = (df['t0'].map(lag_func) | df['t-1'].map(lag_func)).astype(int)

        # выделение временных лагов между одинаковыми gate_id
        for col_name, lag_func in lags.items():
            # (g1 g0 g-1) & (t0 t-1)
            gate_prev = (df['g1'] == df['g0']) & df['t0'].map(lag_func)
            gate_next = (df['g0'] == df['g-1']) & df['t-1'].map(lag_func)
            df[col_name.replace('lag', 'dbl')] = (gate_prev | gate_next).astype(int)

        # группировки для подсчета кол-ва --------------------------------
        grp_month = df.groupby(['month'], as_index=False).agg(
            counts=('timestamp', 'count'),
            user_id_unique=('user_id', lambda x: x.nunique()),
            date_unique=('date', lambda x: x.nunique())
        )
        grp_month['prs'] = grp_month['counts'] / grp_month['counts'].sum()

        grp_week = df.groupby(['week'], as_index=False).agg(
            counts=('timestamp', 'count'),
            user_id_unique=('user_id', lambda x: x.nunique())
        )
        grp_week['prs'] = grp_week['counts'] / grp_week['counts'].sum()

        grp_date = df.groupby(['date'], as_index=False).agg(
            date_cnt=('timestamp', 'count')
        )
        grp_gate = df.groupby(['date', 'gate_id'], as_index=False).agg(
            gate_cnt=('timestamp', 'count'),
        )
        # группировки -------------------------------------------------

        # это реализовано в initial_preparation более красиво
        # df = df.merge(grp_date, on=['date'], how='left')
        # df = df.merge(grp_gate, on=['date', 'gate_id'], how='left')

        if model_columns is None:
            model_columns = df.columns.to_list()

        if "user_id" not in model_columns:
            model_columns.insert(0, "user_id")

        self.train_idxs = df[df.month.isin(self.train_months)].index
        self.valid_idxs = df[df.month.isin(self.valid_months)].index

        df = self.cat_dummies(df)

        df = self.apply_scaler(df)

        model_columns.extend(self.new_columns)

        exclude_columns = [col for col in self.exclude_columns if col in df.columns]
        exclude_columns.extend(col for col in df.columns if col not in model_columns)

        if exclude_columns:
            df.drop(exclude_columns, axis=1, inplace=True)

        self.exclude_columns = exclude_columns

        # Переводим типы данных в минимально допустимые - экономим ресурсы
        df = memory_compression(df)

        return df

    def fit_transform(self, df, file_df=None, out_five_percent=False,
                      remake_gates_mask=False, model_columns=None):
        """
        fit + transform
        :param df: исходный ФД
        :param file_df: Предобработанный Файл .pkl с полным путём
        :param out_five_percent: граница 5% при оптределении выбросов
        :param remake_gates_mask: получить шаблоны масок из трейна, иначе взять из класса
        :param model_columns: список колонок, которые будут использованы в модели
        :return: ДФ с фичами
        """
        df = self.fit(df, file_df=file_df, out_five_percent=out_five_percent,
                      remake_gates_mask=remake_gates_mask)
        df = self.transform(df, model_columns=model_columns)
        return df

    def train_test_split(self, df, y=None, *args, **kwargs):
        """
        Деление на обучающую и валидационную выборки
        :param df: ДФ
        :param y: целевая переменная
        :param args: аргументы
        :param kwargs: именованные аргументы
        :return: x_train, x_valid, y_train, y_valid
        """
        if any(key in kwargs for key in ('train_size', 'test_size')):
            if 'test_size' in kwargs:
                train_size = 1 - kwargs['test_size']
            else:
                train_size = kwargs['train_size']
            if train_size > 0.99:
                train_size = 0.99
            train_rows = int(len(df) * train_size)
            x_train = df.iloc[:train_rows]
            x_valid = df.iloc[train_rows:]
            if y is None:
                y_train = y_valid = None
            else:
                y_train = y.iloc[:train_rows]
                y_valid = y.iloc[train_rows:]
        else:

            x_train = df.loc[self.train_idxs]
            x_valid = df.loc[self.valid_idxs]
            if y is None:
                y_train = y_valid = None
            else:
                y_train = y.loc[self.train_idxs]
                y_valid = y.loc[self.valid_idxs]

            self.comment.update(train_months=self.train_months,
                                valid_months=self.valid_months)

        return x_train, x_valid, y_train, y_valid

    @staticmethod
    def make_sample(df, days=7):
        """
        Для опытов оставим небольшой сэмпл из данных и виде первых дней days
        :param df: ДФ
        :param days: количество дней для обучения и +1 день для теста, чтобы код не падал
        :return: ДФ сеэмпла данных
        """
        dates = sorted(df['date'].unique())[:days + 1]
        temp = df[df['date'].isin(dates)]
        temp.loc[temp['date'] == dates[-1], 'user_id'] = -1
        return temp

    def train_valid_split(self, df, test_size=0.2, SEED=17, drop_outlets=False):
        """
        Деление на обучающую и валидационную выборки
        :param df: ДФ
        :param test_size: test_size
        :param SEED: SEED
        :param drop_outlets: удалить редких юзеров и гейты
        :return: x_train, x_valid, y_train, y_valid
        """
        train = df.drop(['user_id'], axis=1)
        target = df['user_id']

        # Split the train_df into training and testing sets
        X_train, X_valid, y_train, y_valid = train_test_split(train, target,
                                                              test_size=test_size,
                                                              stratify=target,
                                                              random_state=SEED)
        if drop_outlets:
            if 'gate_id' in df.columns:
                outlet_user_gate = df.user_id.isin([4, 51, 52]) | df.gate_id.isin([0, 16])
                self.comment.update(drop_users='4,51,52', drop_gates='0,16')
            else:
                outlet_user_gate = df.user_id.isin([4, 51, 52])
                self.comment.update(drop_users='4,51,52')
            outlet_index = df[outlet_user_gate].index
            X_train = X_train[~X_train.index.isin(outlet_index)]
            X_valid = X_valid[~X_valid.index.isin(outlet_index)]
            y_train = y_train[~y_train.index.isin(outlet_index)]
            y_valid = y_valid[~y_valid.index.isin(outlet_index)]

        return X_train, X_valid, y_train, y_valid


if __name__ == "__main__":
    cat_columns = ['gate_id', 'weekday', 'hour']

    file_dir = Path(__file__).parent
    file_train = file_dir.joinpath('train.csv')
    file_test = file_dir.joinpath('test.csv')

    data_cls = DataTransform(category_columns=cat_columns)
    data_cls.preprocess_path_file = file_dir.joinpath('preprocess_df_.pkl')
    # data_cls.exclude_columns = ['timestamp', 'date']

    train_df = pd.read_csv(file_train, parse_dates=['timestamp'], index_col='row_id')
    test_df = pd.read_csv(file_test, parse_dates=['timestamp'], index_col='row_id')
    test_df.insert(0, 'user_id', -1)

    all_df = pd.concat([train_df, test_df])
    # print(all_df.info())

    test_size = 0.2
    SEED = 17

    X_train, X_valid, y_train, y_valid = data_cls.train_valid_split(train_df,
                                                                    test_size=test_size,
                                                                    SEED=SEED,
                                                                    drop_outlets=True)
    print(data_cls.comment)
