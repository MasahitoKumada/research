import seaborn as sns
import matplotlib.pyplot as plt


def check_predict(pred, oof, filepath):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.tight_layout()
    sns.distplot(pred, label='Test Predict')
    sns.distplot(oof, label='Out Of Fold')
    ax.legend()
    ax.grid()
    fig.savefig(filepath)
    plt.close()