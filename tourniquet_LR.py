from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

import optuna

from data_process_tourniquet import DataTransform
from data_process_tourniquet import predict_train_valid, predict_test, get_max_num
from print_time import print_time, print_msg

__import__("warnings").filterwarnings('ignore')

file_logs = Path(r'D:\python-txt\tourniquet\scores_local.logs')
max_num = get_max_num(file_logs)

start_time = print_msg('Обучение логистической регрессии...')

file_dir = Path(__file__).parent
file_train = file_dir.joinpath('train.csv')
file_test = file_dir.joinpath('test.csv')

train_df = pd.read_csv(file_train, parse_dates=['timestamp'], index_col='row_id')
test_df = pd.read_csv(file_test, parse_dates=['timestamp'], index_col='row_id')
test_df.insert(0, 'user_id', -1)

all_df = pd.concat([train_df, test_df])

numeric_columns = ['min', 'seconds', 'beep_count', 'beep_gate']

cat_columns = ['gate_id', 'weekday',
               'hour',
               # 'is_weekend',
               # '1day', '2day', 'last_day-1', 'last_day',
               # 'DofY1'
               ]

data_cls = DataTransform(category_columns=cat_columns, drop_first=False,
                         # numeric_columns=numeric_columns, scaler=StandardScaler,
                         )

# prefix_preprocess = '_min_0'
# prefix_preprocess = '_fp'
# prefix_preprocess = '_fp2'
prefix_preprocess = '_MV2'
data_cls.preprocess_path_file = file_dir.joinpath(f'preprocess_df{prefix_preprocess}.pkl')

data_cls.exclude_columns = [
    'timestamp', 'date', 'time', 'day', 'week', 'month',
    'dayofweek',
    # 'min',
    'sec',
    'minutes',
    # 'seconds',
    '3day', 'last_days',
    # 'DofY1',
    # 'DofY2',
    # 'lag0', 'lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag8', 'lag9',
    # 'dbl0', 'dbl1', 'dbl2', 'dbl3', 'dbl4', 'dbl5', 'dbl6', 'dbl7', 'dbl8', 'dbl9',
    # 'lag8',
    # 'dbl8',
    'lag9',
    'dbl9',
    'm1', 'm2', 'm3', 'm4', 'd1', 'd2', 'd3', 'd4', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6',
    'g5', 'g4', 'g3', 'g2', 'g1', 'g0', 'g-1', 'g-2', 'g-3', 'g-4', 'g-5',
    't5', 't4', 't3', 't2', 't1', 't0', 't-1', 't-2', 't-3', 't-4', 't-5',
    'G7_3_3_11', 'G7_3_10', 'G8_8_5', 'G11_6_6_5', 'G11_11_10', 'G13_13_6_6',  # 2 и 4
    'G6_6_7', 'G7_8_8', 'G11_11_4_4_9', 'G13_13_12_12_11', 'G15_3_3_10_11',  # 5
    # 'G-1_-1_10', 'G-1_-1_11', 'G3_3_4', 'G3_3_10_11_6', 'G3_10_11_6', 'G13_13_11',  # 6
]

data_cls.beep_outlet = 98.7
all_df = data_cls.fit(all_df, file_df=data_cls.preprocess_path_file, remake_gates_mask=True)

# # для опытов оставим небольшой сэмпл из данных и виде первых дней days=5
# all_df = data_cls.make_sample(all_df, days=2)

all_df = data_cls.transform(all_df)
print(all_df.info())

train_df = all_df[all_df.user_id > -1]
test_df = all_df[all_df.user_id < 0]

model_columns = train_df.columns.to_list()
model_columns.remove('user_id')

print('Обучаюсь на колонках:', model_columns)
print('Категорийные колонки:', cat_columns)
print('Исключенные колонки:', data_cls.exclude_columns)

# добавление user_id для валидации кто попался только один раз
for user_id in train_df.user_id.unique():
    if len(train_df.loc[train_df.user_id == user_id]) < 4:
        print(f'user_id = {user_id}')
        train_df = train_df.append(train_df.loc[train_df.user_id == user_id])

print(f'Размер train_df = {train_df.shape}, test_df = {test_df.shape}')

train = train_df.drop(['user_id'], axis=1)
target = train_df['user_id']
test_df = test_df.drop(['user_id'], axis=1)

test_sizes = (0.2,)
# test_sizes = (0.2, 0.25)
# test_sizes = np.linspace(0.15, 0.25, 11)
for test_size in test_sizes:
# for iters in range(50, 201, 10):
    max_num += 1
    test_size = 0.2
    SEED = 17
    iters = 200

    print(f'test_size: {test_size}')
    # Split the train_df into training and testing sets
    X_train, X_valid, y_train, y_valid = train_test_split(train, target,
                                                          test_size=test_size,
                                                          stratify=target,
                                                          random_state=SEED)

    # # самодельный train_test_split
    # X_train, X_valid, y_train, y_valid = data.train_test_split(train, target)

    num_folds = 4
    skf = StratifiedKFold(n_splits=num_folds, random_state=SEED, shuffle=True)
    split_kf = KFold(n_splits=num_folds, random_state=SEED, shuffle=True)

    fit_on_full_train = True
    use_grid_search = False
    use_optuna_cv = False
    build_model = True
    stratified = True

    clf = LogisticRegression(random_state=SEED,
                             # class_weight='balanced',
                             multi_class="ovr",
                             solver="newton-cholesky",
                             # penalty=None,
                             # penalty='l2',
                             # C=10,
                             # C=c_range,
                             C=8,
                             max_iter=iters,
                             n_jobs=7,
                             )

    DTS = (X_train, X_valid, y_train, y_valid, train, target, test_df)

    if use_grid_search:
        clf = LogisticRegression(random_state=SEED,
                                 multi_class="ovr",
                                 solver="newton-cholesky",
                                 n_jobs=7,
                                 )
        params = {'C': [1, 5, 10, 15],
                  'penalty': [None, "l2"],
                  'class_weight': [None, 'balanced'],
                  # 'solver': ["saga"],
                  'max_iter': [75, 100, 150],
                  }
        gscv = GridSearchCV(estimator=clf, param_grid=params, cv=skf, verbose=50,
                            scoring='f1_weighted', n_jobs=7, refit=True)

        gscv.fit(train, target)

        best_params = gscv.best_params_
        clf = gscv.best_estimator_

    elif use_optuna_cv:
        clf = LogisticRegression(random_state=SEED,
                                 multi_class="ovr",
                                 solver="newton-cholesky",
                                 n_jobs=7,
                                 )
        param_distributions = {
            # "C": optuna.distributions.CategoricalDistribution([1, 5, 10, 15]),
            # "C": optuna.distributions.FloatDistribution(1, 20),
            "C": optuna.distributions.IntDistribution(1, 30),
            # "penalty": optuna.distributions.CategoricalDistribution([None, "l2"]),
            # 'class_weight': optuna.distributions.CategoricalDistribution([None, 'balanced']),
            # 'max_iter': optuna.distributions.CategoricalDistribution([75, 100, 150]),
            # "max_iter": optuna.distributions.IntDistribution(50, 200, step=10),
        }
        optuna_search = optuna.integration.OptunaSearchCV(
            clf, param_distributions, n_trials=99, timeout=600, verbose=2,
            # enable_pruning=True,
        )
        optuna_search.fit(train, target)

        print("Best trial:")
        trial = optuna_search.study_.best_trial

        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        clf = optuna_search
        clf.fit(X_train, y_train)

    else:

        clf.fit(X_train, y_train)

    best_params = {'clf_iters': clf.get_params()['max_iter'],
                   'C': clf.get_params().get('C')
                   }

    print(clf.get_params())

    if build_model:
        if not fit_on_full_train:
            predict_scores = predict_test(0, clf, DTS, max_num, submit_prefix='lr_')

        else:
            predict_scores = predict_test('pool', clf, DTS, max_num, submit_prefix='lr_')
            acc_train, acc_valid, acc_full, roc_auc, f1w = predict_scores
            score = acc_valid

            comment = {'times': prefix_preprocess, 'test_size': test_size, 'SEED': SEED,
                       'size': 'pool'}
            comment.update(data_cls.comment)
            comment.update({'stratified': stratified})
            comment.update(clf.get_params())

            with open(file_logs, mode='a') as log:
                # log.write('num;mdl;roc_auc;acc_train;acc_valid;acc_full;score;WF1;'
                #           'model_columns;exclude_columns;cat_columns;comment\n')
                log.write(f'{max_num};lr;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
                          f'{acc_full:.6f};'
                          f'{score:.6f};{f1w:.6f};{train_df.columns.tolist()};'
                          f'{data_cls.exclude_columns};{cat_columns};{comment}\n')

            # обучение на всем трейне
            print('Обучаюсь на всём трейне...')

            clf.fit(train, target)
            predict_scores = predict_test('full', clf, DTS, max_num, submit_prefix='lr_')

    best_params.update(clf.get_params())
    print('best_params:', best_params)

    if build_model:
        acc_train, acc_valid, acc_full, roc_auc, f1w = predict_scores
        score = acc_valid

        print(f'Weighted F1-score = {f1w:.6f}')
        print('Параметры модели:', clf.get_params())

        print_time(start_time)

        comment = {'times': prefix_preprocess,
                   'test_size': test_size,
                   'SEED': SEED,
                   'clf_iters': clf.get_params()['max_iter'],
                   'C': clf.get_params().get('C'),
                   'stratified': stratified}
        comment.update(data_cls.comment)
        comment.update(clf.get_params())

        with open(file_logs, mode='a') as log:
            # log.write('num;mdl;roc_auc;acc_train;acc_valid;acc_full;score;WF1;'
            #           'model_columns;exclude_columns;cat_columns;comment\n')
            log.write(f'{max_num};lr;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
                      f'{acc_full:.6f};'
                      f'{score:.6f};{f1w:.6f};{train_df.columns.tolist()};'
                      f'{data_cls.exclude_columns};{cat_columns};{comment}\n')
