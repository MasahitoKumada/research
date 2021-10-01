import os
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import japanize_matplotlib
from lib.lightgbm import fit_lgbm
from lib.xgboost import fit_xgb
from lib.svm import fit_svm
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
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgbm
from sklearn.svm import SVC

# 警告文非表示
warnings.resetwarnings()
warnings.filterwarnings('ignore')

# 入出力
INPUT_DIR = "./input/apo"
TRAIN_FILE, TEST_FILE = "train_add_features.csv", "test_add_features.csv"
OUTPUT_DIR = "./output/apo"
OUTPUT_FILENAME = "predict.csv"


# Hold out or K-fold out
IS_Kfold_Out = False
if IS_Kfold_Out:
    # k-cross validation
    FOLD_TYPE = "k-fold" # 'k-fold' or 'k-stratified'
    N_SPLITS = 5


# モデルの特徴重要度の可視化
XGB_FEATURE_FIG, XGB_FEATURE_FILENAME = "feature_importance_xgb.png", "feature_xgb.csv"
LGBM_FEATURE_FIG, LGBM_FEATURE_FILENAME = "feature_importance_lgbm.png", "feature_lgbm.csv"
SVM_FEATURE_FIG, SVM_FEATURE_FILENAME = "feature_importance_svm.png", "feature_svm.csv"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"

# shap
IS_XGB_SHAP = True
IS_LGBM_SHAP = True
IS_SVM_SHAP = True

# 特徴重要度の観察から特徴量削除カラム
XGB_COLUMNS_NAME = []
LGBM_COLUMNS_NAME = []
SVM_COLUMNS_NAME = []

# 再現性
random.seed(1)
np.random.seed(1)


def read_csv(dir, filename):
    file_path = os.path.join(dir, filename)
    return pd.read_csv(file_path)


def split_dataset(df):

    select_clms = [ 
        'Score', 'Druggability Score', 'Number of Alpha Spheres',
        'Total SASA', 'Polar SASA', 'Apolar SASA', 'Volume',
        'Mean local hydrophobic density', 'Mean alpha sphere radius',
        'Mean alp. sph. solvent access', 'Apolar alpha sphere proportion',
        'Hydrophobicity score', 'Volume score', 'Polarity score',
        'Charge score', 'Proportion of polar atoms', 'Alpha sphere density',
        'Cent. of mass - Alpha Sphere max dist', 'Flexibility',
        'ALA', 'ARG', 'ASN','ASP','CYS','GLN','GLU','GLY',
        'ILE', 'LEU', 'LYS','MET','PHE','PRO','SER','THR',
        'TRP', 'TYR', 'VAL', 'HIS', 'ASX', 'GLX', 'UNK' ]


    return df[select_clms], df['cryptic pocket flag'], df['PDB Name']


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
    # plt.close()


def main():

    # read input file
    # for train
    train_df = read_csv(INPUT_DIR, TRAIN_FILE)
    X, y, X_train_pdb_name= split_dataset(train_df)

    # for test
    test_df = read_csv(INPUT_DIR, TEST_FILE)
    X_test, y_test, X_test_pdb_name= split_dataset(test_df)

    # select k-fold type
    if IS_Kfold_Out:
        fold = select_fold_type(FOLD_TYPE)
        cv = list(fold.split(X, y)) # もともとが generator なため明示的に list に変換する

    # train param setting
    hyper_params = json.load(open('./lib/hyper_param.json', 'r'))
    xgb_params = hyper_params['xgb']
    lgbm_params = hyper_params['lgbm']
    svm_params = hyper_params['svm']

    ## for train
    # xgboost
    xgb_X = X.copy()
    # 特徴重要度の観察から特徴量削除
    xgb_X_droped = xgb_X.drop(columns=XGB_COLUMNS_NAME)
    xgb_models = []
    if IS_Kfold_Out:
        # Grid Search
        xgb_gs_cv = GridSearchCV(
                            xgb.XGBClassifier(), # 識別器
                            xgb_params, # 最適化したいパラメータセット 
                            cv=cv, # 交差検定の回数
                            scoring='neg_mean_squared_error',
                            verbose=1,
                            return_train_score = True
                        )
        xgb_gs_cv.fit(xgb_X_droped, y)
        xgb_best_param = xgb_gs_cv.best_params_
        
        print('XgBoost Best parameter: {}'.format(xgb_best_param))

        xgb_oof, xgb_models = fit_xgb(xgb_X_droped.values, y, cv, params=xgb_best_param)
    else:
        # Grid Search
        xgb_gs_cv = GridSearchCV(
                            xgb.XGBClassifier(), # 識別器
                            xgb_params, # 最適化したいパラメータセット 
                            cv=2, # 交差検定の回数
                            scoring='neg_mean_squared_error',
                            verbose=1,
                            return_train_score = True
                        )
        xgb_gs_cv.fit(xgb_X_droped, y)
        xgb_best_param = xgb_gs_cv.best_params_

        print('XgBoost Best parameter: {}'.format(xgb_best_param))

        xgb_model = xgb.XGBClassifier(**xgb_best_param)
        xgb_model.fit(xgb_X_droped, y)
        xgb_models.append(xgb_model)

    
    # lightgbm
    lgbm_X = X.copy()
    # 特徴重要度の観察から特徴量削除
    lgbm_X_droped = lgbm_X.drop(columns=LGBM_COLUMNS_NAME)
    lgbm_models = []

    if IS_Kfold_Out:
        # Grid Search
        lgbm_gs_cv = GridSearchCV(
                            lgbm.LGBMClassifier(), # 識別器
                            lgbm_params, # 最適化したいパラメータセット 
                            cv=cv, # 交差検定の回数
                            scoring='neg_mean_squared_error',
                            verbose=1,
                            return_train_score = True
                        )
        lgbm_gs_cv.fit(lgbm_X_droped, y)
        lgbm_best_param = lgbm_gs_cv.best_params_
        
        print('LightGBM Best parameter: {}'.format(lgbm_best_param))

        lgbm_oof, lgbm_models = fit_lgbm(lgbm_X_droped.values, y, cv, params=lgbm_best_param)

    else:

        # Grid Search
        lgbm_gs_cv = GridSearchCV(
                            lgbm.LGBMClassifier(), # 識別器
                            lgbm_params, # 最適化したいパラメータセット
                            cv=2, # 交差検定の回数
                            scoring='neg_mean_squared_error',
                            verbose=1,
                            return_train_score = True
                        )
        lgbm_gs_cv.fit(lgbm_X_droped, y)
        lgbm_best_param = lgbm_gs_cv.best_params_

        print('LightGBM Best parameter: {}'.format(lgbm_best_param))


        lgbm_model = lgbm.LGBMClassifier(**lgbm_best_param)
        lgbm_model.fit(lgbm_X_droped, y)
        lgbm_models.append(lgbm_model)


    # svm
    # 標準化 & 特徴重要度の観察から特徴量削除
    stdsc = StandardScaler().fit(X.copy())
    svm_X_droped = pd.DataFrame(stdsc.transform(X.copy()), columns=X.copy().columns).drop(columns=SVM_COLUMNS_NAME)
    svm_models = []

    if IS_Kfold_Out:
        # Grid Search
        svm_gs_cv = GridSearchCV(
                            SVC(), # 識別器
                            svm_params, # 最適化したいパラメータセット 
                            cv=cv, # 交差検定の回数
                            scoring='neg_mean_squared_error',
                            verbose=1,
                            return_train_score = True
                        )
        svm_gs_cv.fit(svm_X_droped, y)
        svm_best_param = svm_gs_cv.best_params_

        print('Support Vector Machine Best parameter: {}'.format(svm_best_param))

        svm_oof, svm_models = fit_svm(svm_X_droped.values, y, cv, params=svm_best_param)
        
    else:

        # Grid Search
        svm_gs_cv = GridSearchCV(
                            SVC(), # 識別器
                            svm_params, # 最適化したいパラメータセット
                            cv=2, # 交差検定の回数 
                            scoring='neg_mean_squared_error',
                            verbose=1,
                            return_train_score = True
                        )
        svm_gs_cv.fit(svm_X_droped, y)
        svm_best_param = svm_gs_cv.best_params_


        svm_model = SVC(**svm_best_param, probability=True)
        svm_model.fit(svm_X_droped, y)
        svm_models.append(svm_model)



    ## for test
    ## for xgb
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
        shap = Shap(df_test_droped_xgb, xgb_models, X_test_pdb_name, 'xgboost')
        shap.summary_plot(os.path.join(OUTPUT_DIR, './shap/xgb/shap_summary_xgb.png'))
        shap.decision_plot(os.path.join(OUTPUT_DIR, './shap/xgb/shap_decision_xgb.png'))
        shap.decision_ok_vs_miss_plot(xgb_pred, y_test, os.path.join(OUTPUT_DIR, './shap/xgb/shap_decision_ok_vs_miss_xgb.png'))
        shap.decision_miss_data_plot(xgb_pred, y_test, os.path.join(OUTPUT_DIR, './shap/xgb/shap_decision_miss_xgb.png'))
        shap.decision_high_prob_data_plot(xgb_pred, 0.90, os.path.join(OUTPUT_DIR, './shap/xgb/shap_decision_ok_high_prob_xgb.png'))
        shap.dependence_plot(ind='Mean alp. sph. solvent access', interaction_index='Polarity score', out_path=os.path.join(OUTPUT_DIR, './shap/xgb/shap_dependence_xgb.png'))
        shap.force_plot(xgb_pred, y_test, os.path.join(OUTPUT_DIR, './shap/xgb/shap_force_miss_xgb.png'))


    ## for lgbm
    df_test_droped_lgbm = X_test.copy().drop(columns=LGBM_COLUMNS_NAME)
    if IS_Kfold_Out:
        pred = np.array([model.predict(df_test_droped_lgbm.values) for model in lgbm_models])
        lgbm_pred = np.mean(pred, axis=0)
        lgbm_pred_proba = np.mean(np.array([model.predict_proba(df_test_droped_lgbm.values) for model in lgbm_models]), axis=0)
    else:
        lgbm_pred = lgbm_model.predict(df_test_droped_lgbm.values)
        lgbm_pred_proba = lgbm_model.predict_proba(df_test_droped_lgbm.values)

    # probaability
    lgbm_pred_proba = np.mean(np.array([model.predict_proba(df_test_droped_lgbm.values) for model in lgbm_models]), axis=0)

    # shap for lgbm
    if IS_LGBM_SHAP:
        shap = Shap(df_test_droped_lgbm, lgbm_models, X_test_pdb_name, 'lightgbm')
        shap.summary_plot(os.path.join(OUTPUT_DIR, './shap/lgbm/shap_summary_lgbm.png'))
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
        shap.summary_plot(os.path.join(OUTPUT_DIR, './shap/svm/shap_summary_svm.png'))
        shap.decision_plot(os.path.join(OUTPUT_DIR, './shap/svm/shap_decision_svm.png'))
        shap.decision_ok_vs_miss_plot(svm_pred, y_test, os.path.join(OUTPUT_DIR, './shap/svm/shap_decision_ok_vs_miss_svm.png'))
        shap.decision_miss_data_plot(svm_pred, y_test, os.path.join(OUTPUT_DIR, './shap/svm/shap_decision_miss_svm.png'))
        shap.decision_high_prob_data_plot(svm_pred, 0.90, os.path.join(OUTPUT_DIR, './shap/svm/shap_decision_ok_high_prob_svm.png'))
        shap.dependence_plot(ind='Mean alp. sph. solvent access', interaction_index='Polarity score', out_path=os.path.join(OUTPUT_DIR, './shap/svm/shap_dependence_svm.png'))
        shap.force_plot(svm_pred, y_test, os.path.join(OUTPUT_DIR, './shap/svm/shap_force_miss_svm.png'))


    ## ensamble
    (xgb_ratio, lgbm_ratio, svm_ratio)=(0.5, 0.5, 0.0)
    assert xgb_ratio + lgbm_ratio + svm_ratio == 1.0
    y_pred = xgb_ratio * xgb_pred + lgbm_ratio * lgbm_pred + svm_ratio * svm_pred
    y_pred = np.where(y_pred < 0, 0, np.round(y_pred).astype(int))
    score = f1_score(y_test, y_pred) * 100

    print('Y true: ', y_test.tolist())
    print('XgBoost predict: ', xgb_pred)
    print('LightGBM predict: ', lgbm_pred)
    print('SVM predict: ', svm_pred)

    pred_proba = xgb_ratio * xgb_pred_proba + lgbm_ratio * lgbm_pred_proba + svm_ratio * svm_pred_proba 
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


    #混合行列作成
    make_confusion_matrix(y_test, xgb_pred, CONFUSION_MATRIX_FILENAME + 'xgb')
    make_confusion_matrix(y_test, lgbm_pred, CONFUSION_MATRIX_FILENAME + 'lgbm')
    make_confusion_matrix(y_test, svm_pred, CONFUSION_MATRIX_FILENAME + 'svm')
    make_confusion_matrix(y_test, y_pred, CONFUSION_MATRIX_FILENAME + 'ensamble')



    # 予測がまともに動いているかどうかチェック
    if IS_Kfold_Out:
        # 予測がまともに動いているかどうかチェック
        check_predict(xgb_pred, xgb_oof, os.path.join(OUTPUT_DIR, "check_predict_xgb_.png"))
        check_predict(lgbm_pred, lgbm_oof, os.path.join(OUTPUT_DIR, "check_predict_lgbm.png"))
        check_predict(svm_pred, svm_oof, os.path.join(OUTPUT_DIR, "check_predict_svm.png"))



if __name__ == "__main__":
    main()