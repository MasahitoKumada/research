import os
import sys
import pandas as pd
from contextlib import contextmanager
from time import time
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from lib.lightgbm import fit_lgbm
from lib.xgboost import fit_xgb
from lib.visualize_importance import visualize_importance
from lib.chech_predict import chech_predict
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


INPUT_DIR = "./input"
TRAIN_FILE, TEST_FILE = "train.csv", "test.csv"
OUTPUT_DIR = "./output"
OUTPUT_FILENAME = "predict.csv"

XGB_FEATURE_FIG, XGB_FEATURE_FILENAME = "feature_importance_xgb.png", "feature_xgb.csv"
LGBM_FEATURE_FIG, LGBM_FEATURE_FILENAME = "feature_importance_lgbm.png", "feature_lgbm.csv"

FOLD_TYPE = 'k-stratified' # 'k-ford' or 'k-stratified'
N_SPLITS = 2

# 特徴重要度の観察から特徴量削除カラム
XGB_COLUMNS_NAME = []
LGBM_COLUMNS_NAME = []


np.random.seed(1)



def read_csv(dir, filename):
    file_path = os.path.join(dir, filename)
    return pd.read_csv(file_path)


def split_train_data_first(df):
    y = df[df.columns[0]].values
    X = df[df.columns[1:]]
    return y, X

def split_train_data_last(df):
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]].values
    return X, y


def main():

    # read input file
    # for train
    train_df = read_csv(INPUT_DIR, TRAIN_FILE)
    X, y = split_train_data_last(train_df)
    _, X = split_train_data_first(X)

    # for test
    test_df = read_csv(INPUT_DIR, TEST_FILE)
    X_test, y_test = split_train_data_last(test_df)
    X_test_pdb_name, X_test = split_train_data_first(X_test)

    fold = None
    if FOLD_TYPE=='k-fold':
        # k-fold setting
        fold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=71)
    # 層化k分割検証(k-stratified)
    elif FOLD_TYPE=='k-stratified':
      fold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=11)

    cv = list(fold.split(X, y)) # もともとが generator なため明示的に list に変換する


    # train param setting
    xgb_params = {
        'objective': 'binary:logistic', # 最小化させるべき損失関数を指定する.
        'eval_metric': 'logloss', # 検証を行うためのデータの評価指標.
        'max_depth': 10, # 木の深さの最大値過学習を制御するために用いられる.高いと過学習しやすくなる.
        'learning_rate': 0.1,
        'subsample': 0.7, # 各木においてランダムに抽出される標本の割合小さくすることで, 過学習を避けることができるが保守的なモデルとなる.
        'gamma':0.7, # 分割が、損失関数の減少に繋がる場合にのみノードの分割を行う. モデルをより保守的にする.
        'n_estimators': 10000,
        'importance_type': 'gain'  # 特徴重要度計算のロジック(後述)
        }

    lgbm_params = {
        'objective': 'binary', # 目的関数. これの意味で最小となるようなパラメータを探します. 
        'metrics': 'binary_logloss', # multi_logloss(softmax関数)とmulti_error(正答率)の2つ．
        'learning_rate': 0.1, # 学習率. 小さいほどなめらかな決定境界が作られて性能向上に繋がる場合が多いです、がそれだけ木を作るため学習に時間がかかります
        'max_depth': 6, # 木の深さ. 深い木を許容するほどより複雑な交互作用を考慮するようになります.
        'num_leaves':31, # 木モデルの複雑さを制御するための主要なパラメータ. 2^(max_depth)よりも小さくする必要があります.
        'n_estimators': 10000, # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
        'colsample_bytree': 0.5, # 木を作る際に考慮する特徴量の割合. 1以下を指定すると特徴をランダムに欠落させます。小さくすることで, まんべんなく特徴を使うという効果があります.
        'importance_type': 'gain' # 特徴重要度計算のロジック(後述)
        }

    # train
    # xgboost
    xgb_X = X.copy()
    #  特徴重要度の観察から特徴量削除
    xgb_X_droped = xgb_X.drop(columns=XGB_COLUMNS_NAME)

    xgb_oof, xgb_models = fit_xgb(xgb_X_droped.values, y, cv, params=xgb_params)
    # 特徴重要度の確認
    fig, ax = visualize_importance(xgb_models, xgb_X_droped, os.path.join(OUTPUT_DIR, XGB_FEATURE_FIG) , os.path.join(OUTPUT_DIR, XGB_FEATURE_FILENAME))

    # lightgbm
    lgbm_X = X.copy()
    #  特徴重要度の観察から特徴量削除
    lgbm_X_droped = lgbm_X.drop(columns=LGBM_COLUMNS_NAME)

    lgbm_oof, lgbm_models = fit_lgbm(lgbm_X_droped.values, y, cv, params=lgbm_params)
    # 特徴重要度の確認
    fig, ax = visualize_importance(lgbm_models, lgbm_X_droped, os.path.join(OUTPUT_DIR, LGBM_FEATURE_FIG), os.path.join(OUTPUT_DIR, LGBM_FEATURE_FILENAME))


    (xgb_ratio, lgbm_ratio)=(0.5, 0.5)
    assert xgb_ratio+lgbm_ratio == 1.0

    # for xgb
    df_test_droped_xgb = X_test.copy().drop(columns=XGB_COLUMNS_NAME)
    pred = np.array([model.predict(df_test_droped_xgb.values) for model in xgb_models])
    xgb_pred = np.mean(pred, axis=0)

    # for lgbm
    df_test_droped_lgbm = X_test.copy().drop(columns=LGBM_COLUMNS_NAME)
    pred = np.array([model.predict(df_test_droped_lgbm.values) for model in lgbm_models])
    lgbm_pred = np.mean(pred, axis=0)

    # ansamble
    pred = xgb_ratio * xgb_pred + lgbm_ratio * lgbm_pred
    pred = np.where(pred < 0, 0, np.round(pred).astype(int))

    score = f1_score(y_test, pred, average='macro') * 100

    pred_df = pd.DataFrame({
        "PDB Name": X_test_pdb_name,
        "cryptic pocket flag True": y_test,
        "cryptic pocket flag predict": pred,
        "score": score
        })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME), index=False)

    # 予測がまともに動いているかどうかチェック
    chech_predict(xgb_pred, xgb_oof, os.path.join(OUTPUT_DIR, "chech_predict_xgb_.png"))
    chech_predict(lgbm_pred, lgbm_oof, os.path.join(OUTPUT_DIR, "chech_predict_lgbm.png"))



if __name__ == "__main__":
    main()