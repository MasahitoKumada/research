import os
import pandas as pd

ORIGINAL_DATA_PATH = "./data_files/"
INPUT_FILE_HOLO_PATH, OUTPUT_FILE_HOLO_PATH = "./family_index2_holo.csv", os.path.join(ORIGINAL_DATA_PATH, "holo/input")
INPUT_FILE_APO_PATH, OUTPUT_FILE_APO_PATH = "./family_index2_apo.csv", os.path.join(ORIGINAL_DATA_PATH, "apo/input")

def main():
    # read holo cdv
    dataset_holo_df = pd.read_csv(INPUT_FILE_HOLO_PATH)
    
    for i in range(len(dataset_holo_df['PDBid'])):
      pdbid = dataset_holo_df['PDBid'].iloc[i]
      origin_pdb_path = ORIGINAL_DATA_PATH + pdbid + ".pdb1"
      order = "cp " + origin_pdb_path + " " + OUTPUT_FILE_HOLO_PATH
      os.system(order)

    # rename
    order = "rename s/\.pdb1$/.pdb/" + " " + os.path.join(OUTPUT_FILE_HOLO_PATH, "*.pdb1")
    os.system(order)

    # read apo cdv 
    dataset_apo_df = pd.read_csv(INPUT_FILE_APO_PATH)
    
    for i in range(len(dataset_apo_df['PDBid'])):
      pdbid = dataset_apo_df['PDBid'].iloc[i]
      origin_pdb_path = ORIGINAL_DATA_PATH + pdbid + ".pdb1"
      order = "cp " + origin_pdb_path + " " + OUTPUT_FILE_APO_PATH
      os.system(order)
    
    # rename
    order = "rename s/\.pdb1$/.pdb/" + " " + os.path.join(OUTPUT_FILE_APO_PATH, "*.pdb1")
    os.system(order)

    print("finish.")


if __name__ == '__main__':
    main()
