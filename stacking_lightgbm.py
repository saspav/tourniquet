"""StackingClassifier"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

import lightgbm as lg
import optuna

from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

from data_process_tourniquet import DataTransform, PREDICTIONS_DIR
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
        "num_boost_round": trial.suggest_int("num_boost_round", 500, 900, step=50),
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


def merge_files(name_files):
    df = pd.DataFrame()
    for filename in name_files:
        print('Обрабатываю файл:', Path(filename).name)
        temp = pd.read_csv(filename, index_col='row_id')
        col_prefix = Path(filename).name[:2]
        temp.rename(columns={col: f'{col_prefix}_{col}' for col in temp.columns if
                             col not in ['user_id']}, inplace=True)
        if len(df):
            if 'user_id' in temp.columns:
                temp.drop('user_id', axis=1, inplace=True)
            df = df.merge(temp, how='left', left_index=True, right_index=True)
        else:
            df = temp
    return df


file_logs = Path(r'D:\python-txt\tourniquet\scores_local.logs')
max_num = get_max_num(file_logs)

start_time = print_msg('Стекинг классификаторов...')

stacking_dir = PREDICTIONS_DIR.joinpath('stacking')
files_train = glob(f'{stacking_dir}/*train*.csv')
files_test = glob(f'{stacking_dir}/*proba*.csv')

train_df = merge_files(files_train)
test_df = merge_files(files_test)

model_columns = [col for col in train_df.columns.to_list() if 'target' not in col]
train_df = train_df[model_columns]
model_columns.remove('user_id')

test_df = test_df[[col for col in test_df.columns.to_list() if 'target' not in col]]

print(f'Размер train_df = {train_df.shape}, test_df = {test_df.shape}')

# train_df = train_df.iloc[:100, :]

train = train_df.drop(['user_id'], axis=1)
target = train_df['user_id']

prefix_preprocess = 'stack'
cat_columns = []
data_cls = DataTransform(use_catboost=True, category_columns=cat_columns)
data_cls.exclude_columns = []
data_cls.comment = {}

print('Обучаюсь на колонках:', model_columns)
print('Категорийные колонки:', cat_columns)
print('Исключенные колонки:', data_cls.exclude_columns)

test_sizes = (0.2,)
# test_sizes = (0.2, 0.25)
# test_sizes = np.linspace(0.15, 0.25, 11)
# for num_iters in range(400, 901, 50):
# for SEED in range(100):
for test_size in test_sizes:

    max_num += 1

    test_size = 0.2
    num_iters = 600
    SEED = 17

    print(f'test_size: {test_size} SEED={SEED}')

    # Split the train_df into training and testing sets
    X_train, X_valid, y_train, y_valid = train_test_split(train, target,
                                                          test_size=test_size,
                                                          stratify=target,
                                                          random_state=SEED)

    pool_train = lg.Dataset(data=X_train, label=y_train, free_raw_data=False)
    pool_valid = lg.Dataset(data=X_valid, label=y_valid, free_raw_data=False)

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
                predict_scores = predict_test(idx, clf, DTS, max_num, submit_prefix='st_',
                                              save_predict_proba=False)
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
                predict_scores = predict_test(0, clf, DTS, max_num, submit_prefix='st_',
                                              save_predict_proba=False)

            else:
                predict_scores = predict_test('pool', clf, DTS, max_num, submit_prefix='st_',
                                              save_predict_proba=False)
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
                    log.write(f'{max_num};st;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
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

                model.fit(train, target, verbose=50)
                predict_scores = predict_test('full', model, DTS, max_num,
                                              submit_prefix='st_', save_predict_proba=False)

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
            log.write(f'{max_num};st;{roc_auc:.6f};{acc_train:.6f};{acc_valid:.6f};'
                      f'{acc_full:.6f};'
                      f'{score:.6f};{f1w:.6f};{train_df.columns.tolist()};'
                      f'{data_cls.exclude_columns};{cat_columns};{comment}\n')
