import os 
import optuna
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_contour
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_slice
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgbm
from sklearn.svm import SVC


def objective_variable(model_type, X_train, y_train, X_test, y_test):
    def objective(trial):

        if model_type=='RandomForest':
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 16),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 4),
                'max_features': trial.suggest_int("max_features", 2, X_test.shape[1]),
                "random_state": 1
            }
            clf = RandomForestClassifier(**params)

        elif model_type=='XGboost':
            params = {
                "objective": "binary:logistic",
                "eval_metric": "rmse", 
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "subsample": trial.suggest_float("subsample", 0.1, 0.9),
                "gamma": trial.suggest_float("gamma", 0.1, 0.9),
                "n_estimators": trial.suggest_int("n_estimators", 1000, 7000),
                "importance_type": "gain"
            }
            clf = xgb.XGBClassifier(**params)
            
        elif model_type=='LightGBM':
            params = {

                "objective": "binary",
                "eval_metric": "l2", 
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 3, 60),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "n_estimators": trial.suggest_int("n_estimators", 1000, 7000),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 0.9),
                "importance_type": "gain"
            }
            clf = lgbm.LGBMClassifier(**params)

        elif model_type=='SVM':
            params = {
                "C": trial.suggest_loguniform("C", 1e-5, 1),
                "gamma": trial.suggest_loguniform("gamma", 1e-5, 1),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "sigmoid"]),
                "decision_function_shape": "ovr",
            }
            clf = SVC(**params, probability=True)

        # score_funcs = ['f1']
        # scores = cross_validate(clf, X_train, y_train, cv=4, scoring=score_funcs)
        # return scores['test_f1'].mean()

        clf.fit(X_train, y_train)
        return f1_score(y_test, clf.predict(X_test))

    return objective


def param_tuning(model_type, X_train, y_train, X_test, y_test, n_trials=10):
    # Optuna
    # directionで目的変数の最大化を指定する.
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=1))
    study.optimize(objective_variable(model_type, X_train, y_train, X_test, y_test), n_trials=n_trials)
    return study


def vizualize_tuning_result(study, output_dir, model_type):
    # 最適化の様子の視覚化
    fig = plot_optimization_history(study)
    fig.write_image((os.path.join(output_dir, 'optim_history_{}.png'.format(model_type))))

    # ハイパーパラメータ間の関係の視覚化
    fig = plot_contour(study)
    fig.write_image((os.path.join(output_dir, 'contour_{}.png'.format(model_type))))

    # ハイパーパラメータの重要度の視覚化
    fig = plot_param_importances(study)
    fig.write_image((os.path.join(output_dir, 'param_importances_{}.png'.format(model_type))))

    # パラメータの食い合わせと目的変数の結果の視覚化
    fig = plot_parallel_coordinate(study)
    fig.write_image((os.path.join(output_dir, 'parallel_coordinate_{}.png'.format(model_type))))

    # 各パラメータの値と目的変数の結果視覚化
    fig = plot_slice(study)
    fig.write_image((os.path.join(output_dir, 'slice_{}.png'.format(model_type))))