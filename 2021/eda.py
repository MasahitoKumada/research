import os
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from scipy import stats


IN_DIR = './data'
IN_TRAIN_FILE, IN_TEST_FILE = 'cryptic_pocket_apo_train_add_features.csv', 'cryptic_pocket_apo_test_add_features.csv'
OUT_DIR = './eda'

os.makedirs(OUT_DIR, exist_ok=True)

IS_HISTGRAM = True
IS_INSPECTION = True

if IS_HISTGRAM:
    OUT_FILE_HIST = 'cryptic_site_vs_other_site_for_apo_add_features_eda_hist'
    OUT_FILE_VIOLIN = 'cryptic_site_vs_other_site_for_apo_add_features_eda_violin'
    N_COLS = 5



def read_csv(dir, filename):
    file_path = os.path.join(dir, filename)
    return pd.read_csv(file_path)


def multi_plot(df, filename):

    columns = df.columns
    n_figs = len(columns)
    n_rows = n_figs // N_COLS + 1
    
    fig, axes = plt.subplots(ncols=N_COLS, nrows=n_rows, figsize=(N_COLS * 3, n_rows * 3))
    fig.subplots_adjust(wspace=0.7, hspace=1.4)

    # プロット
    for c, ax in zip(columns, axes.ravel()):
            if df[c].dtypes=='object':
                df[c].value_counts().plot(ax=ax, kind='bar')
            else:
                df[c].plot(ax=ax, kind='hist')

            ax.set_title(c)

    fig.tight_layout()
    # save figure
    plt.savefig(os.path.join(OUT_DIR, filename))


def vs_multi_plot(df1, df2, filename):

    columns = df1.columns
    n_figs = len(columns)
    n_rows = n_figs // N_COLS 
    
    fig, axes = plt.subplots(ncols=N_COLS, nrows=n_rows, figsize=(N_COLS * 8, n_rows * 6))
    fig.subplots_adjust(wspace=0.8, hspace=1.4)

    # プロット
    for c, ax in zip(columns, axes.ravel()):


            if df1[c].dtypes=='object':
                df1[c].value_counts().plot(ax=ax, kind='bar', alpha=0.5, label='cryptic site', color='C2', range=(min(df1[c].value_counts()), max(df1[c].value_counts())), bins=15)
                df2[c].value_counts().plot(ax=ax, kind='bar', alpha=0.5, label='concave site', color='C3', range=(min(df2[c].value_counts()), max(df2[c].value_counts())), bins=15)
            else:
                df1[c].plot(ax=ax, kind='hist', alpha=0.5, label='cryptic site', color='C2', range=(min(df1[c]), max(df1[c])), bins=15)
                df2[c].plot(ax=ax, kind='hist', alpha=0.5, label='concave site', color='C3', range=(min(df2[c]), max(df2[c])), bins=15)

            ax.set_title(c)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    fig.tight_layout()
    # save figure
    plt.savefig(os.path.join(OUT_DIR, filename))


def violin_plot(df, filename):

    plt.style.use('default')
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set3')

    columns = df.columns
    n_figs = len(columns)
    n_rows = n_figs // N_COLS
    
    fig, axes = plt.subplots(ncols=N_COLS, nrows=n_rows, figsize=(N_COLS * 3, n_rows * 3))
    fig.subplots_adjust(wspace=0.7, hspace=1.4)

    # プロット
    for c, ax in zip(columns, axes.ravel()):
            sns.violinplot(x='cryptic pocket flag', y=c, data=df, 
                hue='cryptic pocket flag', dodge=False, jitter=True, 
                color='black', palette='Set1', ax=ax)

            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    fig.tight_layout()
    # save figure
    plt.savefig(os.path.join(OUT_DIR, filename))



def two_sample_test(input_df):

    test_dic = {} 
    for clm in input_df.columns:
        cryptic_site = input_df[input_df['cryptic pocket flag']==1]
        other_site = input_df[input_df['cryptic pocket flag']==0]

        if not clm=="cryptic pocket flag":
            test_dic[clm] = stats.ks_2samp(cryptic_site[clm], other_site[clm]).pvalue

    test_dic_sorted = sorted(test_dic.items(), key=lambda x:x[1])
    for key,val in dict(test_dic_sorted).items():
        print('culum name: {}, p value: {}'.format(key, val))


def main():

    input_train_df = read_csv(IN_DIR, IN_TRAIN_FILE)
    input_test_df = read_csv(IN_DIR, IN_TEST_FILE)
    input_df = pd.concat([input_train_df, input_test_df], axis=0)

    if IS_HISTGRAM :
        cryptic_site_apo_df = input_df[input_df['cryptic pocket flag']==1].drop(columns='PDB Name')
        other_site_apo_df = input_df[input_df['cryptic pocket flag']==0].drop(columns='PDB Name')
        vs_multi_plot(cryptic_site_apo_df, other_site_apo_df, OUT_FILE_HIST)
        violin_plot(input_df.drop(columns='PDB Name'), OUT_FILE_VIOLIN)

    if IS_INSPECTION:
        two_sample_test(input_df.drop(columns='PDB Name'))

    print('OK.')


if __name__ == "__main__":
    main()