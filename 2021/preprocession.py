import os
import pandas as pd
from sklearn.model_selection import train_test_split


IN_PATH = "./data"
OUT_PATH = "./input"


def read_csv(filename):
    return pd.read_csv(os.path.join(IN_PATH, filename))


def df_concat(df1, df2, axis):
    return pd.concat([df1, df2], axis=axis) 


def save_csv(df, filename):
    df.to_csv(os.path.join(OUT_PATH, filename), index=False)


def main():

    # read
    input_df = read_csv("cryptic_pocket1.csv")
    cryptic_pocket_df = input_df[input_df["cryptic pocket flag"]==1]
    normal_pocket_df = input_df[input_df["cryptic pocket flag"]==0]
    
    # split
    cryptic_pocket_train, cryptic_pocket_test = train_test_split(cryptic_pocket_df, train_size=0.8, test_size=0.2)
    normal_pocket_train, normal_pocket_test = train_test_split(normal_pocket_df, train_size=0.8, test_size=0.2)

    # concat
    train_df = df_concat(cryptic_pocket_train, normal_pocket_train, axis=0)
    test_df = df_concat(cryptic_pocket_test, normal_pocket_test, axis=0)

    # save
    save_csv(train_df, "train.csv")
    save_csv(test_df, "test.csv")




if __name__ == "__main__":
    main()