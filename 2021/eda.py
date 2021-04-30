import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import numpy as np


IN_DIR, IN_FILE = './data', 'cryptic_pocket2.csv'
OUT_DIR = './eda'
OUT_FILE = 'cryptic_site_vs_other_site_for_apo_eda.png'
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
                df1[c].value_counts().plot(ax=ax, kind='bar', label='cryptic site', color="C2")
                df2[c].value_counts().plot(ax=ax, kind='bar', label='concave surface patches', color="C3")
            else:
                df1[c].plot(ax=ax, kind='hist', label='cryptic site', color="C2")
                df2[c].plot(ax=ax, kind='hist',  label='concave surface patches', color="C3")

            ax.set_title(c)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    fig.tight_layout()
    # save figure
    plt.savefig(os.path.join(OUT_DIR, filename))



def main():

    input_df = read_csv(IN_DIR, IN_FILE)

    # print(input_df)
    cryptic_site_apo_df = input_df[input_df["cryptic pocket flag"]==1].drop(columns='PDB Name')
    other_site_apo_df = input_df[input_df["cryptic pocket flag"]==0].drop(columns='PDB Name')

    vs_multi_plot(cryptic_site_apo_df, other_site_apo_df, OUT_FILE)


if __name__ == "__main__":
    main()