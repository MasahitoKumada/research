import numpy as np
from sklearn.svm import SVC
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


def fit_svm(X, y, cv, params: dict=None):
    """support vector machine を CrossValidation の枠組みで学習を行なう function"""

    # パラメータがないときは、空の dict で置き換える
    if params is None:
        params = {}

    models = []
    # training data の target と同じだけのゼロ配列を用意
    # float にしないと悲しい事件が起こるのでそこだけ注意
    oof_pred = np.zeros_like(y, dtype=np.float)

    print('----SVM train start----')

    for i, (idx_train, idx_valid) in enumerate(cv): 
        # この部分が交差検証のところです。データセットを cv instance によって分割します
        # training data を trian/valid に分割
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = SVC(**params, probability=True)
        with timer(prefix='fit fold={} '.format(i + 1)):
            clf.fit(x_train, y_train)
            pred_i = clf.predict(x_valid)
            pred_i = np.where(pred_i < 0, 0, pred_i)
            oof_pred[idx_valid] = pred_i

        models.append(clf)

        # print(pred_i)
        print(f"Fold {i+1} F1: {f1_score(y_valid, pred_i) * 100}")

    score = f1_score(y, oof_pred) * 100
    print('FINISHED | SVM Whole F1: {:.4f}'.format(score))
    print()
    return oof_pred, models