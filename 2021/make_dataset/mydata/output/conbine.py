import pandas as pd
import warnings
import sys

warnings.resetwarnings()
warnings.simplefilter('ignore', RuntimeError)

# inpuut
INPUT_APO_TRAIN_PATH, INPUT_APO_TEST_PATH  = './cryptic_pocket_apo_train.csv', 'cryptic_pocket_apo_test.csv'
INPUT_APO_TRAIN_ADD_FEATURES_PATH, INPUT_APO_TEST_ADD_FEATURES_PATH = "./train_apo2_add_features.csv", "./test_apo2_add_features.csv"

# output
OUTPUT_APO_TRAIN_PATH, OUTPUT_APO_TEST_PATH = './cryptic_pocket_apo_train_add_features.csv', 'cryptic_pocket_apo_test_add_features.csv'

def main():
    # read csv
    apo_tarin_df = pd.read_csv(INPUT_APO_TRAIN_PATH)
    apo_test_df = pd.read_csv(INPUT_APO_TEST_PATH)
    apo_tarin_add_features_df = pd.read_csv(INPUT_APO_TRAIN_ADD_FEATURES_PATH)
    apo_test_add_features_df = pd.read_csv(INPUT_APO_TEST_ADD_FEATURES_PATH)

    # merge csv
    train_df = pd.merge(apo_tarin_df, apo_tarin_add_features_df, left_on=['Apo in paper', 'Pocket番号'], right_on=['PDB Name', 'Pocket num'], how='left').drop(['Apo in paper', 'Pocket番号'], axis=1)
    test_df = pd.merge(apo_test_df, apo_test_add_features_df, left_on=['Apo in paper', 'Pocket番号'], right_on=['PDB Name', 'Pocket num'], how='left').drop(['Apo in paper', 'Pocket番号'], axis=1)


    a = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR','TRP', 'TYR', 'VAL', 'HIS','ASX', 'GLX', 'UNK']
    print(train_df[train_df['PDB Name']=='1PKLB'][a])

    # save csv
    train_df.to_csv(OUTPUT_APO_TRAIN_PATH, index=False)
    test_df.to_csv(OUTPUT_APO_TEST_PATH, index=False)

    print('OK.')


if __name__ == '__main__':
    main()
