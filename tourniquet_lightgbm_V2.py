import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import seaborn as sns

import lightgbm as lg
import optuna

from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from data_process_tourniquet import DataTransform
from data_process_tourniquet import predict_train_valid, predict_test, get_max_num
from print_time import print_time, print_msg

__import__("warnings").filterwarnings('ignore')


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    trial_params = {
        "objective": "multiclass",
        "num_class": target.max() + 1,
        "metric": "multi_logloss",
        "seed": SEED,
        "verbosity": -1,
        # "num_boost_round": 600,
        "num_boost_round": trial.suggest_int("num_boost_round", 500, 650, step=50),
        # "boosting_type": "goss",
        # "boosting_type": trial.suggest_categorical("boosting_type",
        #                                            ["gbdt", "dart", "goss"]),
        "class_weight": None,
        # "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        # "is_unbalance": True,
        "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),
        # "learning_rate": 0.01,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03, step=0.005),
        # "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03, step=0.01),

        # "depth": trial.suggest_int("depth", 2, 12),
        # "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        # "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        # "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        # "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        # "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        # "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        # "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
    gbm = lg.train(trial_params, pool_train, valid_sets=[pool_valid],
                   callbacks=[lg.early_stopping(50), pruning_callback])

    pred_valid = gbm.predict(X_valid)
    pred_labels = np.argmax(pred_valid, axis=1)
    accuracy = accuracy_score(y_valid, pred_labels)
    return accuracy


file_logs = Path(r'D:\python-txt\tourniquet\scores_local.logs')
max_num = get_max_num(file_logs)

start_time = print_msg('Обучение lightGBM классификатор...')

file_dir = Path(__file__).parent
file_train = file_dir.joinpath('train.csv')
file_test = file_dir.joinpath('test.csv')

train_df = pd.read_csv(file_train, parse_dates=['timestamp'], index_col='row_id')
test_df = pd.read_csv(file_test, parse_dates=['timestamp'], index_col='row_id')
test_df.insert(0, 'user_id', -1)

all_df = pd.concat([train_df, test_df])

cat_columns = ['gate_id', 'weekday',
               'hour',
               # 'is_weekend',
               # '1day', '2day', 'last_day-1', 'last_day',
               # 'DofY1'
               ]

data_cls = DataTransform(use_catboost=True,
                         # numeric_columns=numeric_columns,
                         category_columns=cat_columns,
                         )

prefix_preprocess = '_min_0'
# prefix_preprocess = '_fp'
# prefix_preprocess = '_fp2'
# prefix_preprocess = '_MV2'
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
    # 'G6_6_7', 'G7_8_8', 'G11_11_4_4_9', 'G13_13_12_12_11', 'G15_3_3_10_11',  # 5
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
# for num_iters in range(500, 701, 50):
# for SEED in range(100):
for test_size in test_sizes:
# for num_leaves in range(20, 51, 5):
    max_num += 1

    test_size = 0.2
    num_iters = 600
    SEED = 17

    print(f'test_size: {test_size} SEED={SEED}')

    # Split the train_df into training and testing sets
    # X_train, X_valid, y_train, y_valid = train_test_split(train, target,
    #                                                       test_size=test_size,
    #                                                       stratify=target,
    #                                                       random_state=SEED)

    X_train, X_valid, y_train, y_valid = data_cls.train_valid_split(train_df,
                                                                    test_size=test_size,
                                                                    SEED=SEED,
                                                                    drop_outlets=False)

    # # самодельный train_test_split
    # X_train, X_valid, y_train, y_valid = data_cls.train_test_split(train, target)

    pool_train = lg.Dataset(data=X_train, label=y_train, free_raw_data=False,
                            feature_name=model_columns,
                            categorical_feature=cat_columns)
    pool_valid = lg.Dataset(data=X_valid, label=y_valid, free_raw_data=False,
                            feature_name=model_columns,
                            categorical_feature=cat_columns)

    num_folds = 5
    skf = StratifiedKFold(n_splits=num_folds)
    split_kf = KFold(n_splits=num_folds)

    fit_on_full_train = False
    use_grid_search = False
    use_cv_folds = False
    build_model = True
    stratified = True

    models, models_scores, predict_scores = [], [], []

    clf_params = dict(objective="multiclass",
                      num_class=target.nunique(),
                      # class_weight='balanced',
                      is_unbalance=True,
                      learning_rate=0.01,
                      num_boost_round=num_iters,
                      # num_leaves=num_leaves,
                      # num_leaves=63,
                      # max_depth=10,
                      # max_depth=5,
                      # seed=42,
                      seed=SEED,
                      early_stopping_round=50,
                      n_jobs=7,
                      # verbose=-1,
                      verbose=50,
                      # device="gpu",
                      # gpu_platform_id=0,
                      # gpu_device_id=0,
                      )

    clf = lg.LGBMClassifier(**clf_params)

    if use_grid_search:
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
            direction="minimize",
        )
        study.optimize(objective, n_trials=40)

        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        best_params = trial.params

        clf_params.update(best_params)
        print('clf_params', clf_params)

        clf = lg.LGBMClassifier(**clf_params)

    if use_cv_folds:
        if stratified:
            skf_folds = skf.split(train, target)
        else:
            skf_folds = split_kf.split(train)

        for idx, (train_idx, valid_idx) in enumerate(skf_folds, 1):
            print(f'Фолд {idx} из {num_folds}')
            X_train = train.iloc[train_idx]
            X_valid = train.iloc[valid_idx]
            y_train = target.iloc[train_idx]
            y_valid = target.iloc[valid_idx]

            clf = lg.LGBMClassifier(**clf_params)

            clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                    eval_metric='multi_logloss',
                    )
            models.append(clf)
            if build_model:
                DTS = (X_train, X_valid, y_train, y_valid, train, target, test_df)
                predict_scores = predict_test(idx, clf, DTS, max_num)
                models_scores.append(predict_scores)
                acc_train, acc_valid, acc_full, roc_auc, f1w = predict_scores
                score = acc_valid
                comment = {'times': prefix_preprocess,
                           'test_size': test_size,
                           'SEED': SEED,
                           'size': f'pool_{idx}'}
                comment.update(data_cls.comment)
                comment.update({'stratified': stratified})
                comment.update(clf.get_params())

                with open(file_logs, mode='a') as log:
                    # log.write('num;mdl;roc_auc;acc_train;acc_valid;acc_full;score;WF1;'
                    #           'model_columns;exclude_columns;cat_columns;comment\n')
                    log.write(f'{max_num};lg;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
                              f'{acc_full:.6f};'
                              f'{score:.6f};{f1w:.6f};{train_df.columns.tolist()};'
                              f'{data_cls.exclude_columns};{cat_columns};{comment}\n')

        best_params = {'iterations': [clf.best_iteration_ for clf in models]}

    else:
        DTS = (X_train, X_valid, y_train, y_valid, train, target, test_df)

        clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                eval_metric='multi_logloss',
                )

        print(clf.best_iteration_)
        print(clf.get_params())

        best_params = {'clf_iters': clf.best_iteration_,
                       'clf_lr': clf.get_params()['learning_rate']}

        models.append(clf)

        if build_model:
            if not fit_on_full_train:
                predict_scores = predict_test(0, clf, DTS, max_num)

            else:
                predict_scores = predict_test('pool', clf, DTS, max_num)
                acc_train, acc_valid, acc_full, roc_auc, f1w = predict_scores
                score = acc_valid
                comment = {'times': prefix_preprocess,
                           'test_size': test_size,
                           'SEED': SEED,
                           'size': 'pool'}
                comment.update(data_cls.comment)
                comment.update({'stratified': stratified})
                comment.update(models[0].get_params())

                with open(file_logs, mode='a') as log:
                    # log.write('num;mdl;roc_auc;acc_train;acc_valid;acc_full;score;WF1;'
                    #           'model_columns;exclude_columns;cat_columns;comment\n')
                    log.write(f'{max_num};lg;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
                              f'{acc_full:.6f};'
                              f'{score:.6f};{f1w:.6f};{train_df.columns.tolist()};'
                              f'{data_cls.exclude_columns};{cat_columns};{comment}\n')

                print('Обучаюсь на всём трейне...')
                model = lg.LGBMClassifier(objective="multiclass",
                                          num_class=target.nunique(),
                                          seed=SEED,
                                          early_stopping_round=50,
                                          # device_type='cuda',
                                          # device_type='gpu',
                                          )

                model.fit(train, target, verbose=50, cat_features=cat_columns)
                predict_scores = predict_test('full', model, DTS, max_num)

        best_params.update(clf.get_params())

    print('best_params:', best_params)

    if build_model:
        if len(models) > 1:
            predict_scores = [np.mean(arg) for arg in zip(*models_scores)]

        acc_train, acc_valid, acc_full, roc_auc, f1w = predict_scores
        score = acc_valid

        print(f'Weighted F1-score = {f1w:.6f}')
        print('Параметры модели:', clf.get_params())

        print_time(start_time)

        comment = {'times': prefix_preprocess,
                   'test_size': test_size,
                   'SEED': SEED,
                   'clf_iters': models[0].best_iteration_,
                   'clf_lr': models[0].get_params()['learning_rate'],
                   'stratified': stratified}
        comment.update(data_cls.comment)
        comment.update(models[0].get_params())

        with open(file_logs, mode='a') as log:
            # log.write('num;mdl;roc_auc;acc_train;acc_valid;acc_full;score;WF1;'
            #           'model_columns;exclude_columns;cat_columns;comment\n')
            log.write(f'{max_num};lg;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
                      f'{acc_full:.6f};'
                      f'{score:.6f};{f1w:.6f};{train_df.columns.tolist()};'
                      f'{data_cls.exclude_columns};{cat_columns};{comment}\n')
