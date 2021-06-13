import os
import pandas as pd
from contextlib import contextmanager
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import japanize_matplotlib
from lib.lightgbm import fit_lgbm
from lib.xgboost import fit_xgb
from lib.visualize_importance import visualize_importance
from lib.check_predict import check_predict
from lib.shap import Shap
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
import json


INPUT_DIR = "./input_apo"
TRAIN_FILE, TEST_FILE = "train.csv", "test.csv"
OUTPUT_DIR = "./output_apo"
OUTPUT_FILENAME = "predict.csv"

XGB_FEATURE_FIG, XGB_FEATURE_FILENAME = "feature_importance_xgb.png", "feature_xgb.csv"
LGBM_FEATURE_FIG, LGBM_FEATURE_FILENAME = "feature_importance_lgbm.png", "feature_lgbm.csv"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"

FOLD_TYPE = "k-stratified" # 'k-ford' or 'k-stratified'
N_SPLITS = 4

# 特徴重要度の観察から特徴量削除カラム
XGB_COLUMNS_NAME = []
LGBM_COLUMNS_NAME = []

# 再現性
random.seed(1)
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


def select_fold_type(fold_type):
    # k-fold type
    fold = None
    if fold_type=='k-fold':
        # k-fold setting
        fold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=71)
    # 層化k分割検証(k-stratified)
    elif fold_type=='k-stratified':
        fold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=11)
    return fold



def main():

    # select k-fold type
    fold = select_fold_type(FOLD_TYPE)

    # for test
    test_df = read_csv(INPUT_DIR, TEST_FILE)
    X_test, y_test = split_train_data_last(test_df)
    X_test_pdb_name, X_test = split_train_data_first(X_test)
    
    # read input file
    # for train
    train_df = read_csv(INPUT_DIR, TRAIN_FILE)
    X, y = split_train_data_last(train_df)
    X_train_pdb_name, X = split_train_data_first(X)

    cv = list(fold.split(X, y)) # もともとが generator なため明示的に list に変換する

    # train param setting
    hyper_params = json.load(open('./lib/hyper_param.json', 'r'))

    xgb_params = hyper_params['xgb']
    lgbm_params = hyper_params['lgbm']

    # xgb_params = {
    #     'objective': 'binary:logistic', # 最小化させるべき損失関数を指定する.
    #     'eval_metric': 'logloss', # 検証を行うためのデータの評価指標.
    #     'max_depth': 12, # 木の深さの最大値過学習を制御するために用いられる.高いと過学習しやすくなる.
    #     'learning_rate': 0.01,
    #     'subsample': 0.7, # 各木においてランダムに抽出される標本の割合小さくすることで, 過学習を避けることができるが保守的なモデルとなる.
    #     'gamma':0.7, # 分割が、損失関数の減少に繋がる場合にのみノードの分割を行う. モデルをより保守的にする.
    #     'n_estimators': 10000,
    #     'importance_type': 'gain'  # 特徴重要度計算のロジック(後述)
    #     }

    # lgbm_params = {
    #     'objective': 'binary', # 目的関数. これの意味で最小となるようなパラメータを探します. 
    #     'metrics': 'binary_logloss', # multi_logloss(softmax関数)とmulti_error(正答率)の2つ．
    #     'learning_rate': 0.07, # 学習率. 小さいほどなめらかな決定境界が作られて性能向上に繋がる場合が多いです、がそれだけ木を作るため学習に時間がかかります
    #     'max_depth': 6, # 木の深さ. 深い木を許容するほどより複雑な交互作用を考慮するようになります.
    #     'num_leaves':31, # 木モデルの複雑さを制御するための主要なパラメータ. 2^(max_depth)よりも小さくする必要があります.
    #     'n_estimators': 10000, # 木の最大数. early_stopping という枠組みで木の数は制御されるようにしていますのでとても大きい値を指定しておきます.
    #     'colsample_bytree': 0.5, # 木を作る際に考慮する特徴量の割合. 1以下を指定すると特徴をランダムに欠落させます。小さくすることで, まんべんなく特徴を使うという効果があります.
    #     'importance_type': 'gain' # 特徴重要度計算のロジック(後述)
    #     }

    ## for train
    # xgboost
    xgb_X = X.copy()
    # 特徴重要度の観察から特徴量削除
    xgb_X_droped = xgb_X.drop(columns=XGB_COLUMNS_NAME)
    xgb_oof, xgb_models = fit_xgb(xgb_X_droped.values, y, cv, params=xgb_params)
    

    # lightgbm
    lgbm_X = X.copy()
    # 特徴重要度の観察から特徴量削除
    lgbm_X_droped = lgbm_X.drop(columns=LGBM_COLUMNS_NAME)
    lgbm_oof, lgbm_models = fit_lgbm(lgbm_X_droped.values, y, cv, params=lgbm_params)

    (xgb_ratio, lgbm_ratio)=(0.5, 0.5)
    assert xgb_ratio+lgbm_ratio == 1.0

    ## for test
    # for xgb
    df_test_droped_xgb = X_test.copy().drop(columns=XGB_COLUMNS_NAME)
    pred = np.array([model.predict(df_test_droped_xgb.values) for model in xgb_models])
    xgb_pred = np.mean(pred, axis=0)
    # print(xgb_pred)

    # 特徴重要度の確認
    fig, ax = visualize_importance(xgb_models, df_test_droped_xgb, os.path.join(OUTPUT_DIR, XGB_FEATURE_FIG) , os.path.join(OUTPUT_DIR, XGB_FEATURE_FILENAME))

    # probaability
    xgb_pred_proba = np.mean(np.array([model.predict_proba(df_test_droped_xgb.values) for model in xgb_models]), axis=0)

    # shap for xgb
    shap = Shap(df_test_droped_xgb, xgb_models, 'xgboost')
    shap.summary_plot(os.path.join(OUTPUT_DIR, './shap/xgb/shap_summary_xgb.png'))
    shap.decision_plot(os.path.join(OUTPUT_DIR, './shap/xgb/shap_decision_xgb.png'))
    shap.decision_ok_vs_miss_plot(xgb_pred, y_test, os.path.join(OUTPUT_DIR, './shap/xgb/shap_decision_ok_vs_miss_xgb.png'))
    shap.decision_miss_data_plot(xgb_pred, y_test, os.path.join(OUTPUT_DIR, './shap/xgb/shap_decision_miss_xgb.png'))
    shap.decision_high_prob_data_plot(xgb_pred, 0.90, os.path.join(OUTPUT_DIR, './shap/xgb/shap_decision_ok_high_prob_xgb.png'))
    shap.dependence_plot(ind='Mean alp. sph. solvent access', interaction_index='Polarity score', out_path=os.path.join(OUTPUT_DIR, './shap/xgb/shap_dependence_xgb.png'))
    shap.force_plot(xgb_pred, y_test, os.path.join(OUTPUT_DIR, './shap/xgb/shap_force_miss_xgb.png'))


    # for lgbm
    df_test_droped_lgbm = X_test.copy().drop(columns=LGBM_COLUMNS_NAME)
    pred = np.array([model.predict(df_test_droped_lgbm.values) for model in lgbm_models])
    lgbm_pred = np.mean(pred, axis=0)

    # probaability
    lgbm_pred_proba = np.mean(np.array([model.predict_proba(df_test_droped_lgbm.values) for model in lgbm_models]), axis=0)

    # shap for lgbm
    shap = Shap(df_test_droped_lgbm, lgbm_models, 'lightgbm')
    shap.summary_plot(os.path.join(OUTPUT_DIR, './shap/lgbm/shap_summary_lgbm.png'))
    shap.decision_plot(os.path.join(OUTPUT_DIR, './shap/lgbm/shap_decision_lgbm.png'))
    shap.decision_ok_vs_miss_plot(lgbm_pred, y_test, os.path.join(OUTPUT_DIR, './shap/lgbm/shap_decision_ok_vs_miss_lgbm.png'))
    shap.decision_miss_data_plot(lgbm_pred, y_test, os.path.join(OUTPUT_DIR, './shap/lgbm/shap_decision_miss_lgbm.png'))
    shap.decision_high_prob_data_plot(lgbm_pred, 0.80, os.path.join(OUTPUT_DIR, './shap/lgbm/shap_decision_ok_high_prob_lgbm.png'))
    shap.dependence_plot(ind='Mean alp. sph. solvent access', interaction_index='Polarity score', out_path=os.path.join(OUTPUT_DIR, './shap/lgbm/shap_dependence_lgbm.png'))
    shap.force_plot(lgbm_pred, y_test, os.path.join(OUTPUT_DIR, './shap/lgbm/shap_force_miss_lgbm.png'))


    # 特徴重要度の確認
    fig, ax = visualize_importance(lgbm_models, df_test_droped_lgbm, os.path.join(OUTPUT_DIR, LGBM_FEATURE_FIG), os.path.join(OUTPUT_DIR, LGBM_FEATURE_FILENAME))

    ## ensamble
    y_pred = xgb_ratio * xgb_pred + lgbm_ratio * lgbm_pred
    y_pred = np.where(y_pred < 0, 0, np.round(y_pred).astype(int))
    score = f1_score(y_test, y_pred) * 100

    pred_proba = (xgb_pred_proba + lgbm_pred_proba)/2
    # print(pred_proba)

    pred_df = pd.DataFrame({
        "PDB Name": X_test_pdb_name,
        "cryptic pocket flag True": y_test,
        "cryptic pocket flag predict": y_pred,
        "pred proba 0": pred_proba[:,0],
        "pred proba 1": pred_proba[:,1],
        "f1 score": score
        })
    #テストデータに対する予測結果の保存    
    pred_df.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME), index=False)

    #混合行列作成
    ax = plt.subplot()
    cm = confusion_matrix(y_test, y_pred)
    sns.set(font_scale=2.5) # Adjust to fit
    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues')
    # 軸名
    label_font = {'size':'18'}  # Adjust to fit
    ax.set_xlabel('pred', fontdict=label_font)
    ax.set_ylabel('True', fontdict=label_font)
    # title
    title_font = {'size':'21'}  # Adjust to fit
    ax.set_title('Confusion Matrix', fontdict=title_font)
    # ticks
    ax.tick_params(axis='both', which='major', labelsize=16)  # Adjust to fit
    # save figure
    plt.savefig(os.path.join(OUTPUT_DIR, CONFUSION_MATRIX_FILENAME))
    plt.close()

    # 予測がまともに動いているかどうかチェック
    check_predict(xgb_pred, xgb_oof, os.path.join(OUTPUT_DIR, "check_predict_xgb_.png"))
    check_predict(lgbm_pred, lgbm_oof, os.path.join(OUTPUT_DIR, "check_predict_lgbm.png"))


def main2():
    json_open = open('./lib/hyper_param.json', 'r')
    json_load = json.load(json_open)
    print(json_load)
    print(json_load['xgb'])


if __name__ == "__main__":
    main()