import pandas as pd

INPUT_FILE_PATH = "./family_index2.csv"
OUTPUT_HOLO_FILE_PATH, OUTPUT_APO_FILE_PATH = "./family_index2_holo.csv", "./family_index2_apo.csv"

def main():
    # read csv
    dataset_df = pd.read_csv(INPUT_FILE_PATH)
    # make holo or apo dataset
    holoset_df = dataset_df[dataset_df['Type']==1][dataset_df['In_CryptoSite']==0].groupby('Family').first()
    aposet_df = dataset_df[dataset_df['Type']==0][dataset_df['In_CryptoSite']==0].groupby('Family').first()
    # save as csv
    holoset_df.to_csv(OUTPUT_HOLO_FILE_PATH)
    aposet_df.to_csv(OUTPUT_APO_FILE_PATH)


if __name__ == '__main__':
    main()
