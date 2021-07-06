import os
import sys
import pandas as pd
from contextlib import contextmanager
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import japanize_matplotlib
from lib.lightgbm import fit_lgbm
from lib.xgboost import fit_xgb
from lib.svm import fit_svm
from lib.random_forest import fit_rf
import xgboost as xgb
import lightgbm as lgbm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lib.visualize_importance import visualize_importance
from lib.check_predict import check_predict
from lib.shap import Shap
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import json
import warnings


# 警告文非表示
warnings.resetwarnings()
warnings.filterwarnings('ignore')

# 入出力
INPUT_DIR = "./input/apo"
TRAIN_FILE, TEST_FILE = "train.csv", "test.csv"
OUTPUT_DIR = "./output/apo"
OUTPUT_FILENAME = "predict.csv"

# Hold out or K-fold out
IS_Kfold_Out = False
if IS_Kfold_Out:
    # k-cross validation
    FOLD_TYPE = "k-fold" # 'k-fold' or 'k-stratified'
    N_SPLITS = 5

# モデルの特徴重要度の可視化
RF_FEATURE_FIG, RF_FEATURE_FILENAME = "feature_importance_rf.png", "feature_rf.csv"
XGB_FEATURE_FIG, XGB_FEATURE_FILENAME = "feature_importance_xgb.png", "feature_xgb.csv"
LGBM_FEATURE_FIG, LGBM_FEATURE_FILENAME = "feature_importance_lgbm.png", "feature_lgbm.csv"
SVM_FEATURE_FIG, SVM_FEATURE_FILENAME = "feature_importance_svm.png", "feature_svm.csv"
CONFUSION_MATRIX_FILENAME = "confusion_matrix_"

# shap
IS_RF_SHAP = True
IS_XGB_SHAP = True
IS_LGBM_SHAP = True
IS_SVM_SHAP = True


# 特徴重要度の観察から特徴量削除カラム
RF_COLUMNS_NAME = []
XGB_COLUMNS_NAME = []
LGBM_COLUMNS_NAME = []
SVM_COLUMNS_NAME = []

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


def make_confusion_matrix(y_test, y_pred, filename):
    #混合行列作成
    
    ax = plt.subplot()
    plt.tight_layout()
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
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


def main():

    # read input file
    # for train
    train_df = read_csv(INPUT_DIR, TRAIN_FILE)
    X, y = split_train_data_last(train_df)
    X_train_pdb_name, X = split_train_data_first(X)

    # for test
    test_df = read_csv(INPUT_DIR, TEST_FILE)
    X_test, y_test = split_train_data_last(test_df)
    X_test_pdb_name, X_test = split_train_data_first(X_test)

    # select k-fold type
    if IS_Kfold_Out:
        fold = select_fold_type(FOLD_TYPE)
        cv = list(fold.split(X, y)) # もともとが generator なため明示的に list に変換する

    # train param setting
    hyper_params = json.load(open('./lib/hyper_param.json', 'r'))
    rf_params = hyper_params['rf']
    xgb_params = hyper_params['xgb']
    lgbm_params = hyper_params['lgbm']
    svm_params = hyper_params['svm']

    ## for train
    # random forest
    rf_X = X.copy()
    # 特徴重要度の観察から特徴量削除
    rf_X_droped = rf_X.drop(columns=RF_COLUMNS_NAME)
    rf_models = []
    if IS_Kfold_Out:
        rf_oof, rf_models = fit_rf(rf_X_droped.values, y, cv, params=rf_params)
    else: 
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(rf_X_droped, y)
        rf_models.append(rf_model)

    # xgboost
    xgb_X = X.copy()
    # 特徴重要度の観察から特徴量削除
    xgb_X_droped = xgb_X.drop(columns=XGB_COLUMNS_NAME)
    xgb_models = []
    if IS_Kfold_Out:
        xgb_oof, xgb_models = fit_xgb(xgb_X_droped.values, y, cv, params=xgb_params)
    else:
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(xgb_X_droped, y)
        xgb_models.append(xgb_model)
    
    # lightgbm
    lgbm_X = X.copy()
    # 特徴重要度の観察から特徴量削除
    lgbm_X_droped = lgbm_X.drop(columns=LGBM_COLUMNS_NAME)
    lgbm_models = []
    if IS_Kfold_Out:
        lgbm_oof, lgbm_models = fit_lgbm(lgbm_X_droped.values, y, cv, params=lgbm_params)
    else:
        lgbm_model = lgbm.LGBMClassifier(**lgbm_params)
        lgbm_model.fit(lgbm_X_droped, y)
        lgbm_models.append(lgbm_model)

    # svm
    # 標準化 & 特徴重要度の観察から特徴量削除
    stdsc = StandardScaler().fit(X.copy())
    svm_X_droped = pd.DataFrame(stdsc.transform(X.copy()), columns=X.copy().columns).drop(columns=SVM_COLUMNS_NAME)
    svm_models = []
    if IS_Kfold_Out:
        svm_oof, svm_models = fit_svm(svm_X_droped.values, y, cv, params=svm_params)
    else:
        svm_model = SVC(**svm_params, probability=True)
        svm_model.fit(svm_X_droped, y)
        svm_models.append(svm_model)

    ## for test
    # for random forest
    df_test_droped_rf = X_test.copy().drop(columns=RF_COLUMNS_NAME)
    if IS_Kfold_Out:
        pred = np.array([model.predict(df_test_droped_rf.values) for model in rf_models])
        rf_pred = np.mean(pred, axis=0)
        rf_pred_proba = np.mean(np.array([model.predict_proba(df_test_droped_rf.values) for model in rf_models]), axis=0)
    else:
        rf_pred = rf_model.predict(df_test_droped_rf.values)
        rf_pred_proba = rf_model.predict_proba(df_test_droped_rf.values)

    # 特徴重要度の確認
    fig, ax = visualize_importance(rf_models, df_test_droped_rf, os.path.join(OUTPUT_DIR, RF_FEATURE_FIG) , os.path.join(OUTPUT_DIR, RF_FEATURE_FILENAME))

    # shap for rf
    if IS_RF_SHAP:
        shap = Shap(df_test_droped_rf, rf_models, X_test_pdb_name, 'random_forest')
        shap.summary_plot(os.path.join(OUTPUT_DIR, './shap/rf/shap_summary_violin_rf.png'), 'violin')
        shap.summary_plot(os.path.join(OUTPUT_DIR, './shap/rf/shap_summary_bar_rf.png'), 'bar')
        shap.decision_plot(os.path.join(OUTPUT_DIR, './shap/rf/shap_decision_rf.png'))
        shap.decision_ok_vs_miss_plot(rf_pred, y_test, os.path.join(OUTPUT_DIR, './shap/rf/shap_decision_ok_vs_miss_rf.png'))
        shap.decision_miss_data_plot(rf_pred, y_test, os.path.join(OUTPUT_DIR, './shap/rf/shap_decision_miss_rf.png'))
        shap.decision_high_prob_data_plot(rf_pred, 0.80, os.path.join(OUTPUT_DIR, './shap/rf/shap_decision_ok_high_prob_rf.png'))
        shap.dependence_plot(ind='Mean alp. sph. solvent access', interaction_index='Polarity score', out_path=os.path.join(OUTPUT_DIR, './shap/rf/shap_dependence_rf.png'))
        shap.force_plot(rf_pred, y_test, os.path.join(OUTPUT_DIR, './shap/rf/shap_force_miss_rf.png'))

    # for xgb
    df_test_droped_xgb = X_test.copy().drop(columns=XGB_COLUMNS_NAME)
    if IS_Kfold_Out:
        pred = np.array([model.predict(df_test_droped_xgb.values) for model in xgb_models])
        xgb_pred = np.mean(pred, axis=0)
        xgb_pred_proba = np.mean(np.array([model.predict_proba(df_test_droped_xgb.values) for model in xgb_models]), axis=0)
    else:
        xgb_pred = xgb_model.predict(df_test_droped_xgb.values)
        xgb_pred_proba = xgb_model.predict_proba(df_test_droped_xgb.values)

    # 特徴重要度の確認
    fig, ax = visualize_importance(xgb_models, df_test_droped_xgb, os.path.join(OUTPUT_DIR, XGB_FEATURE_FIG) , os.path.join(OUTPUT_DIR, XGB_FEATURE_FILENAME))

    # shap for xgb
    if IS_XGB_SHAP:
        shap = Shap(df_test_droped_xgb, xgb_models, X_test_pdb_name,'xgboost')
        shap.summary_plot(os.path.join(OUTPUT_DIR, './shap/xgb/shap_summary_violin_xgb.png'), 'violin')
        shap.summary_plot(os.path.join(OUTPUT_DIR, './shap/xgb/shap_summary_bar_xgb.png'), 'bar')
        shap.decision_plot(os.path.join(OUTPUT_DIR, './shap/xgb/shap_decision_xgb.png'))
        shap.decision_ok_vs_miss_plot(xgb_pred, y_test, os.path.join(OUTPUT_DIR, './shap/xgb/shap_decision_ok_vs_miss_xgb.png'))
        shap.decision_miss_data_plot(xgb_pred, y_test, os.path.join(OUTPUT_DIR, './shap/xgb/shap_decision_miss_xgb.png'))
        shap.decision_high_prob_data_plot(xgb_pred, 0.80, os.path.join(OUTPUT_DIR, './shap/xgb/shap_decision_ok_high_prob_xgb.png'))
        shap.dependence_plot(ind='Mean alp. sph. solvent access', interaction_index='Polarity score', out_path=os.path.join(OUTPUT_DIR, './shap/xgb/shap_dependence_xgb.png'))
        shap.force_plot(xgb_pred, y_test, os.path.join(OUTPUT_DIR, './shap/xgb/shap_force_miss_xgb.png'))


    # for lgbm
    df_test_droped_lgbm = X_test.copy().drop(columns=LGBM_COLUMNS_NAME)
    if IS_Kfold_Out:
        pred = np.array([model.predict(df_test_droped_lgbm.values) for model in lgbm_models])
        lgbm_pred = np.mean(pred, axis=0)
        lgbm_pred_proba = np.mean(np.array([model.predict_proba(df_test_droped_lgbm.values) for model in lgbm_models]), axis=0)
    else:
        lgbm_pred = lgbm_model.predict(df_test_droped_lgbm.values)
        lgbm_pred_proba = lgbm_model.predict_proba(df_test_droped_lgbm.values)

    # shap for lgbm
    if IS_LGBM_SHAP:
        shap = Shap(df_test_droped_lgbm, lgbm_models, X_test_pdb_name,'lightgbm')
        shap.summary_plot(os.path.join(OUTPUT_DIR, './shap/lgbm/shap_summary_violin_lgbm.png'), 'violin')
        shap.summary_plot(os.path.join(OUTPUT_DIR, './shap/lgbm/shap_summary_bar_lgbm.png'), 'bar')
        shap.decision_plot(os.path.join(OUTPUT_DIR, './shap/lgbm/shap_decision_lgbm.png'))
        shap.decision_ok_vs_miss_plot(lgbm_pred, y_test, os.path.join(OUTPUT_DIR, './shap/lgbm/shap_decision_ok_vs_miss_lgbm.png'))
        shap.decision_miss_data_plot(lgbm_pred, y_test, os.path.join(OUTPUT_DIR, './shap/lgbm/shap_decision_miss_lgbm.png'))
        shap.decision_high_prob_data_plot(lgbm_pred, 0.80, os.path.join(OUTPUT_DIR, './shap/lgbm/shap_decision_ok_high_prob_lgbm.png'))
        shap.dependence_plot(ind='Mean alp. sph. solvent access', interaction_index='Polarity score', out_path=os.path.join(OUTPUT_DIR, './shap/lgbm/shap_dependence_lgbm.png'))
        shap.force_plot(lgbm_pred, y_test, os.path.join(OUTPUT_DIR, './shap/lgbm/shap_force_miss_lgbm.png'))

    # 特徴重要度の確認
    fig, ax = visualize_importance(lgbm_models, df_test_droped_lgbm, os.path.join(OUTPUT_DIR, LGBM_FEATURE_FIG), os.path.join(OUTPUT_DIR, LGBM_FEATURE_FILENAME))

    # for svm
    # 標準化 & 特徴重要度の観察から特徴量削除
    df_test_droped_svm = pd.DataFrame(stdsc.transform(X_test.copy()), columns=X_test.copy().columns).drop(columns=SVM_COLUMNS_NAME)
    if IS_Kfold_Out:
        pred = np.array([model.predict(df_test_droped_svm.values) for model in svm_models])
        svm_pred = np.mean(pred, axis=0)
        svm_pred_proba = np.mean(np.array([model.predict_proba(df_test_droped_svm.values) for model in svm_models]), axis=0)
    else:
        svm_pred = svm_model.predict(df_test_droped_svm.values)
        svm_pred_proba = svm_model.predict_proba(df_test_droped_svm.values)

    
    # shap for svm
    if IS_SVM_SHAP:
        shap = Shap(df_test_droped_svm, svm_models, X_test_pdb_name, 'svm')
        shap.summary_plot(os.path.join(OUTPUT_DIR, './shap/svm/shap_summary_violin_svm.png'), 'violin')
        shap.summary_plot(os.path.join(OUTPUT_DIR, './shap/svm/shap_summary_bar_svm.png'), 'bar')
        shap.decision_plot(os.path.join(OUTPUT_DIR, './shap/svm/shap_decision_svm.png'))
        shap.decision_ok_vs_miss_plot(svm_pred, y_test, os.path.join(OUTPUT_DIR, './shap/svm/shap_decision_ok_vs_miss_svm.png'))
        shap.decision_miss_data_plot(svm_pred, y_test, os.path.join(OUTPUT_DIR, './shap/svm/shap_decision_miss_svm.png'))
        shap.decision_high_prob_data_plot(svm_pred, 0.80, os.path.join(OUTPUT_DIR, './shap/svm/shap_decision_ok_high_prob_svm.png'))
        shap.dependence_plot(ind='Mean alp. sph. solvent access', interaction_index='Polarity score', out_path=os.path.join(OUTPUT_DIR, './shap/svm/shap_dependence_svm.png'))
        shap.force_plot(svm_pred, y_test, os.path.join(OUTPUT_DIR, './shap/svm/shap_force_miss_svm.png'))


    ## ensamble
    # (rf_ratio, xgb_ratio, lgbm_ratio, svm_ratio)=(0.20, 0.25, 0.26, 0.29) # 4つ全て使う
    (rf_ratio, xgb_ratio, lgbm_ratio, svm_ratio)=(0, 0.2, 0.2, 0.6) # 上位3つ使う
    assert rf_ratio + xgb_ratio + lgbm_ratio + svm_ratio == 1.0
    y_pred = rf_ratio * rf_pred + xgb_ratio * xgb_pred + lgbm_ratio * lgbm_pred + svm_ratio * svm_pred
    y_pred = np.where(y_pred < 0, 0, np.round(y_pred).astype(int))
    score = f1_score(y_test, y_pred) * 100

    print('Y true: ', y_test)
    print('Random Forest predict: ', f1_score(y_test, np.round(rf_pred).astype(int)) * 100, rf_pred)
    print('XgBoost predict: ', f1_score(y_test, np.round(xgb_pred).astype(int)) * 100, xgb_pred)
    print('LightGBM predict: ', f1_score(y_test, np.round(lgbm_pred).astype(int)) * 100, lgbm_pred)
    print('SVM predict: ', f1_score(y_test, np.round(svm_pred).astype(int)) * 100, svm_pred)
    print('Final Predict: ', y_pred)
    print('Total Score: ', score)
    print()
    correct_classified = y_test==svm_pred
    mis_classified = y_test!=svm_pred
    print('correct_classified PDB: ', X_test_pdb_name[correct_classified])
    print('miss_classified PDB: ', X_test_pdb_name[mis_classified])

    pred_proba = rf_ratio * rf_pred_proba + xgb_ratio * xgb_pred_proba + lgbm_ratio * lgbm_pred_proba + svm_ratio * svm_pred_proba 

    pred_df = pd.DataFrame({
        "PDB Name": X_test_pdb_name,
        "cryptic pocket flag True": y_test,
        "cryptic pocket flag predict": svm_pred,
        "pred proba 0": pred_proba [:,0],
        "pred proba 1": pred_proba [:,1],
        "f1 score": score
        })
    #テストデータに対する予測結果の保存    
    pred_df.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_FILENAME), index=False)

    #混合行列作成
    make_confusion_matrix(y_test, rf_pred, CONFUSION_MATRIX_FILENAME + 'rf')
    make_confusion_matrix(y_test, xgb_pred, CONFUSION_MATRIX_FILENAME + 'xgb')
    make_confusion_matrix(y_test, lgbm_pred, CONFUSION_MATRIX_FILENAME + 'lgbm')
    make_confusion_matrix(y_test, svm_pred, CONFUSION_MATRIX_FILENAME + 'svm')
    make_confusion_matrix(y_test, y_pred, CONFUSION_MATRIX_FILENAME + 'ensamble')
    
   

    # 予測がまともに動いているかどうかチェック
    if IS_Kfold_Out:
        check_predict(rf_pred, rf_oof, os.path.join(OUTPUT_DIR, "check_predict_rf_.png"))
        check_predict(xgb_pred, xgb_oof, os.path.join(OUTPUT_DIR, "check_predict_xgb_.png"))
        check_predict(lgbm_pred, lgbm_oof, os.path.join(OUTPUT_DIR, "check_predict_lgbm.png"))
        check_predict(svm_pred, svm_oof, os.path.join(OUTPUT_DIR, "check_predict_svm.png"))



if __name__ == "__main__":
    main()