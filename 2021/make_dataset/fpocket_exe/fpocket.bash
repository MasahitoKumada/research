#!/bin/bash

echo "please type 'apo2' or 'holo2'."
# echo "please type 'apo' or 'holo'."

read TYPE # TYPE="apo2" or "holo2"

echo "please type 'train' or 'test'."
# train or test data
read DATA_TYPE

# setting for paper
INPUT_DIR="../mydata/input/${DATA_TYPE}/${TYPE}/"
OUTPUT_DIR="../mydata/output/${DATA_TYPE}/${TYPE}/"

# setting for moad 
# INPUT_DIR="../mydata/input/train/LBSp_dataset/data_files/${TYPE}/input/"
# OUTPUT_DIR="../mydata/input/train/LBSp_dataset/data_files/${TYPE}/output/"


PDBID_STR_NUM=5 # pdbidの長さ 4 or 5


idx=1
for _pdb in `\find $INPUT_DIR -name '*.pdb' -maxdepth 1 -type f`
do
  fpocket -f $_pdb

  if [$PDBID_STR_NUM -eq 4]; then
    _pdbId=${_pdb:`expr ${#_pdb} - 8`:4} # 4文字
  else
    _pdbId=${_pdb:`expr ${#_pdb} - 9`:5} # 5文字
  fi
  
  # echo $_pdbId
  # echo $INPUT_DIR$_pdbId"_out"

  mv $INPUT_DIR$_pdbId"_out" $OUTPUT_DIR
  echo "No.${idx}: ${_pdb} execution is OK."
  idx=`expr $idx + 1`
done
echo "finish."
