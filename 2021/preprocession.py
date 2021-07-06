import os
import pandas as pd
from sklearn.model_selection import train_test_split


# 入力ファイルから学習データ、テストデータを分割する。
DO_SPLIT_DATA_TRAIN_TEST = False
IN_PATH, IN_FILE = "./data", "cryptic_pocket_apo.csv"

# 学習データとテストデータ(別々に用意した)を読込む。
DO_READ_DATA_TRAIN_TEST = True
IN_PATH, IN_TRAIN_FILE = "./data", "cryptic_pocket_apo_train.csv"
IN_PATH, IN_TEST_FILE = "./data", "cryptic_pocket_apo_test.csv"

# 出力ディレクトリ
OUT_PATH = "./input/apo"



def read_csv(filename):
    return pd.read_csv(os.path.join(IN_PATH, filename))


def df_concat(df1, df2, axis):
    return pd.concat([df1, df2], axis=axis) 


def save_csv(df, filename):
    df.to_csv(os.path.join(OUT_PATH, filename), index=False)


def main():

    if DO_SPLIT_DATA_TRAIN_TEST:
        # read
        input_df = read_csv(IN_FILE)
        cryptic_pocket_df = input_df[input_df["cryptic pocket flag"]==1]
        normal_pocket_df = input_df[input_df["cryptic pocket flag"]==0]
        
        # split
        cryptic_pocket_train, cryptic_pocket_test = train_test_split(cryptic_pocket_df, train_size=0.9, test_size=0.1)
        normal_pocket_train, normal_pocket_test = train_test_split(normal_pocket_df, train_size=0.9, test_size=0.1)

        # concat
        train_df = df_concat(cryptic_pocket_train, normal_pocket_train, axis=0)
        test_df = df_concat(cryptic_pocket_test, normal_pocket_test, axis=0)

    if DO_READ_DATA_TRAIN_TEST:
        train_df = read_csv(IN_TRAIN_FILE)
        test_df = read_csv(IN_TEST_FILE)

    # save
    save_csv(train_df, "train.csv")
    save_csv(test_df, "test.csv")




if __name__ == "__main__":
    main()