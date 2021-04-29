import numpy as np
import lightgbm as lgbm
from sklearn.metrics import f1_score
from contextlib import contextmanager
from time import time


@contextmanager
def timer(logger=None, format_str='{:.3f}[s]', prefix=None, suffix=None):
    if prefix: format_str = str(prefix) + format_str
    if suffix: format_str = format_str + str(suffix)
    start = time()
    yield
    d = time() - start
    out_str = format_str.format(d)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)


def fit_lgbm(X, y, cv, params: dict=None, verbose: int=10):
    """lightGBM を CrossValidation の枠組みで学習を行なう function"""

    # パラメータがないときは、空の dict で置き換える
    if params is None:
        params = {}

    models = []
    # training data の target と同じだけのゼロ配列を用意
    # float にしないと悲しい事件が起こるのでそこだけ注意
    oof_pred = np.zeros_like(y, dtype=np.float)

    for i, (idx_train, idx_valid) in enumerate(cv): 
        # この部分が交差検証のところです。データセットを cv instance によって分割します
        # training data を trian/valid に分割
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = lgbm.LGBMClassifier(**params)

        with timer(prefix='fit fold={} '.format(i + 1)):
             clf.fit(x_train, y_train,
             eval_set=[(x_valid, y_valid)],
             early_stopping_rounds=verbose, verbose=verbose)

        pred_i = clf.predict(x_valid)

        pred_i = np.where(pred_i < 0, 0, pred_i)
        oof_pred[idx_valid] = pred_i
        models.append(clf)

        print(f"Fold {i} F1_MACRO: {f1_score(y_valid, pred_i) * 100}")

    score = f1_score(y, oof_pred) * 100
    print('FINISHED | Whole F1_MACRO: {:.4f}'.format(score))
    return oof_pred, models