import seaborn as sns
import matplotlib.pyplot as plt


def chech_predict(pred, oof, filepath):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.distplot(pred, label='Test Predict')
    sns.distplot(oof, label='Out Of Fold')
    ax.legend()
    ax.grid()
    fig.savefig(filepath)