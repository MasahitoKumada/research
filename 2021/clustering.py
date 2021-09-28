import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import umap.umap_ as umap
from scipy.sparse.csgraph import connected_components
import warnings


# 警告文非表示
warnings.resetwarnings()
warnings.filterwarnings('ignore')


INPUT_DIR = './input/apo'
TRAIN_FILE, TEST_FILE = 'train_add_features.csv', 'test_add_features.csv'
OUT_DIR = './output/apo_add_features/clustering'

os.makedirs(OUT_DIR, exist_ok=True)

SELECT_CLMS = [ 
        'Score', 'Druggability Score', 'Number of Alpha Spheres',
        'Total SASA', 'Polar SASA', 'Apolar SASA', 'Volume',
        'Mean local hydrophobic density', 'Mean alpha sphere radius',
        'Mean alp. sph. solvent access', 'Apolar alpha sphere proportion',
        'Hydrophobicity score', 'Volume score', 'Polarity score',
        'Charge score', 'Proportion of polar atoms', 'Alpha sphere density',
        'Cent. of mass - Alpha Sphere max dist', 'Flexibility',
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
        'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR',
        'TRP', 'TYR', 'VAL', 'HIS' ]




def read_csv(dir, filename):
    file_path = os.path.join(dir, filename)
    return pd.read_csv(file_path)


def main():

    # データの読込み
    train_df = read_csv(INPUT_DIR, TRAIN_FILE)
    train_df['data_type'] = 'train'
    test_df = read_csv(INPUT_DIR, TEST_FILE)
    test_df['data_type'] = 'test'

    ## 前処理
    # データ結合
    whole_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    # 標準化
    whole_df[SELECT_CLMS] = whole_df[SELECT_CLMS].apply(lambda x: (x-x.mean())/x.std(), axis=0)

    features_df = dimensional_compression(whole_df, 'pca', is_annotate=False)
    features_df = dimensional_compression(whole_df, 'tsne', is_annotate=False)
    features_df = dimensional_compression(whole_df, 'svd', is_annotate=False)
    features_df = dimensional_compression(whole_df, 'umap', is_annotate=True)


def dimensional_compression(input_df, select_method, is_annotate=False):

    if select_method=='pca':
        #主成分分析
        pca = PCA(random_state=0)

        features = pca.fit_transform(input_df[SELECT_CLMS])
        # 主成分得点
        tmp_features_df = pd.DataFrame(features, columns=["PC{}".format(x + 1) for x in range(len(input_df[SELECT_CLMS].columns))])
        features_df = pd.concat([tmp_features_df, input_df[['PDB Name', 'data_type', 'cryptic pocket flag']]], axis=1)

        # 可視化
        pca_scatter_plot(features_df.query('data_type=="train"'), features_df.query('data_type=="test"'), 'train', 'test', os.path.join(OUT_DIR, 'pca_result_data_type.png'), is_annotate)
        pca_scatter_plot(features_df.query('`cryptic pocket flag`==1'), features_df.query('`cryptic pocket flag`==0'), 'cryptic site', 'concave site', os.path.join(OUT_DIR, 'pca_result_cryptic_site.png'), is_annotate)

        # 寄与率
        pca_result_sending_rate_plot(pca, os.path.join(OUT_DIR, 'pca_result_sending_rate.png'))

        # 観測変数の寄与度をプロットする
        contribution_of_observed_variables_plot(input_df[SELECT_CLMS], pca, os.path.join(OUT_DIR, 'pca_result_contribution_of_observed_variables.png'))

        return features_df
  
    elif select_method=='tsne':
        # t-sne
        tsne = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=1000)
        features = tsne.fit_transform(input_df[SELECT_CLMS])

        tmp_features_df = pd.DataFrame(features, columns = ['TSNE1', 'TSNE2'])
        features_df = pd.concat([tmp_features_df, input_df[['PDB Name', 'data_type','cryptic pocket flag']]], axis=1)

        # 可視化
        tsne_scatter_plot(features_df.query('data_type=="train"'), features_df.query('data_type=="test"'), 'train', 'test', os.path.join(OUT_DIR, 'tsne_result_data_type.png'), is_annotate)
        tsne_scatter_plot(features_df.query('`cryptic pocket flag`==1'), features_df.query('`cryptic pocket flag`==0'), 'cryptic site', 'concave site', os.path.join(OUT_DIR, 'tsne_result_cryptic_site.png'), is_annotate)

        return features_df

    elif select_method=='svd':
        # SVD
        svd = TruncatedSVD(n_components=2, random_state=0, n_iter=1000)
        features = svd.fit_transform(input_df[SELECT_CLMS])

        tmp_features_df = pd.DataFrame(features, columns = ['SVD1', 'SVD2'])
        features_df = pd.concat([tmp_features_df, input_df[['PDB Name', 'data_type','cryptic pocket flag']]], axis=1)

        # 可視化
        svd_scatter_plot(features_df.query('data_type=="train"'), features_df.query('data_type=="test"'), 'train', 'test', os.path.join(OUT_DIR, 'svd_result_data_type.png'), is_annotate)
        svd_scatter_plot(features_df.query('`cryptic pocket flag`==1'), features_df.query('`cryptic pocket flag`==0'), 'cryptic site', 'concave site', os.path.join(OUT_DIR, 'svd_result_cryptic_site.png'), is_annotate)

    elif select_method=='umap':
        # UMAP
        model_umap = umap.UMAP(n_components=2, n_neighbors=3, random_state=0)
        features = model_umap.fit_transform(input_df[SELECT_CLMS])

        tmp_features_df = pd.DataFrame(features, columns = ['UMAP1', 'UMAP2'])
        features_df = pd.concat([tmp_features_df, input_df[['PDB Name', 'data_type','cryptic pocket flag']]], axis=1)

        # 可視化
        umap_scatter_plot(features_df.query('data_type=="train"'), features_df.query('data_type=="test"'), 'train', 'test', os.path.join(OUT_DIR, 'umap_result_data_type_ano.png'), is_annotate)
        umap_scatter_plot(features_df.query('`cryptic pocket flag`==1'), features_df.query('`cryptic pocket flag`==0'), 'cryptic site', 'concave site', os.path.join(OUT_DIR, 'umap_result_cryptic_site_ano.png'), is_annotate)        


    

def pca_scatter_plot(df1, df2, label1, label2, out_path, is_annotate):
    # 第一主成分と第二主成分でプロットする
    plt.figure(figsize=(10, 10))
    # PDB名表示
    if is_annotate:
        for x, y, name in zip(df1['PC1'], df1['PC2'], df1['PDB Name']):
            plt.text(x, y, name)
        for x, y, name in zip(df2['PC1'], df2['PC2'], df2['PDB Name']):
            plt.text(x, y, name)
    # 散布図描画        
    plt.scatter(df1['PC1'], df1['PC2'], alpha=0.8, c='b', label=label1)
    plt.scatter(df2['PC1'], df2['PC2'], alpha=0.8, c='r', label=label2)
    plt.grid()
    plt.xlabel("PC1",fontsize=12)
    plt.ylabel("PC2",fontsize=12)
    plt.legend(loc='upper right', borderaxespad=0, fontsize=12)
    plt.savefig(out_path)
    plt.close()


def pca_result_sending_rate_plot(pca, out_path):
    # 寄与率
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.savefig(out_path)
    plt.close()


def contribution_of_observed_variables_plot(df, pca, out_path):
    plt.figure(figsize=(16, 10))
    # 第一主成分と第二主成分における観測変数の寄与度をプロットする
    for x, y, name in zip(pca.components_[0], pca.components_[1], df.columns):
        plt.text(x, y, name)
    plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()
    plt.savefig(out_path)
    plt.close()


def tsne_scatter_plot(df1, df2, label1, label2, out_path, is_annotate=False):

    plt.figure(figsize=(10, 10))
    # PDB名表示
    if is_annotate:
        for x, y, name in zip(df1['TSNE1'], df1['TSNE2'], df1['PDB Name']):
            plt.text(x, y, name)
        for x, y, name in zip(df2['TSNE1'], df2['TSNE2'], df2['PDB Name']):
            plt.text(x, y, name)
    # 散布図描画 
    plt.scatter(df1['TSNE1'], df1['TSNE2'], alpha=0.8, c='b', label=label1)
    plt.scatter(df2['TSNE1'], df2['TSNE2'], alpha=0.8, c='r', label=label2)
    plt.grid()
    plt.xlabel("TSNE1",fontsize=12)
    plt.ylabel("TSNE2",fontsize=12)
    plt.legend(loc='upper right', borderaxespad=0, fontsize=12)
    plt.savefig(out_path)
    plt.close()


def svd_scatter_plot(df1, df2, label1, label2, out_path, is_annotate=False):

    plt.figure(figsize=(10, 10))
    # PDB名表示
    if is_annotate:
        for x, y, name in zip(df1['SVD1'], df1['SVD2'], df1['PDB Name']):
            plt.text(x, y, name)
        for x, y, name in zip(df2['SVD1'], df2['SVD2'], df2['PDB Name']):
            plt.text(x, y, name)
    # 散布図描画 
    plt.scatter(df1['SVD1'], df1['SVD2'], alpha=0.8, c='b', label=label1)
    plt.scatter(df2['SVD1'], df2['SVD2'], alpha=0.8, c='r', label=label2)
    plt.grid()
    plt.xlabel('SVD1',fontsize=12)
    plt.ylabel('SVD2',fontsize=12)
    plt.legend(loc='upper right', borderaxespad=0, fontsize=12)
    plt.savefig(out_path)
    plt.close()


def umap_scatter_plot(df1, df2, label1, label2, out_path, is_annotate=False):

    plt.figure(figsize=(10, 10))
    # PDB名表示
    if is_annotate:
        for x, y, name in zip(df1['UMAP1'], df1['UMAP2'], df1['PDB Name']):
            plt.text(x, y, name)
        for x, y, name in zip(df2['UMAP1'], df2['UMAP2'], df2['PDB Name']):
            plt.text(x, y, name)
    # 散布図描画 
    plt.scatter(df1['UMAP1'], df1['UMAP2'], alpha=0.8, c='b', label=label1)
    plt.scatter(df2['UMAP1'], df2['UMAP2'], alpha=0.8, c='r', label=label2)
    plt.grid()
    plt.xlabel('UMAP1',fontsize=12)
    plt.ylabel('UMAP2',fontsize=12)
    plt.legend(loc='upper right', borderaxespad=0, fontsize=12)
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    main()