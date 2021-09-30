import glob
import os
import sys
from natsort import natsorted


BASE_PATH = './train/apo2'

TARGET_DIRS = sorted(glob.glob(os.path.join(BASE_PATH, '*')))

OUTPUT_PATH = './' + BASE_PATH.split('/')[-2] + '_' + BASE_PATH.split('/')[-1]  + '_add_features.csv'


def cnt_amino_csv(contents, pdb_name, pocket_num, amino_hist):

    if contents=='':
        header = 'PDB Name,Pocket num,ALA,ARG,ASN,ASP,CYS,GLN,GLU,GLY,ILE,LEU,LYS,MET,PHE,PRO,SER,THR,TRP,TYR,VAL,HIS,ASX,GLX,UNK\n'
        tmp_content = header 
    else:
        tmp_content = '\n'

    tmp_content += pdb_name + ',' + pocket_num + ','

    for key, val in amino_hist.items():
        # print(key, val)
        tmp_content += str(val) + ','

    contents += tmp_content[:-1]

    return contents


def cnt_amino_acid(amino_info):

    amino_hist = {
        'ALA': 0, 'ARG': 0, 'ASN': 0, 'ASP': 0,
        'CYS': 0, 'GLN': 0, 'GLU': 0, 'GLY': 0,
        'ILE': 0, 'LEU': 0, 'LYS': 0, 'MET': 0,
        'PHE': 0, 'PRO': 0, 'SER': 0, 'THR': 0,
        'TRP': 0, 'TYR': 0, 'VAL': 0, 'HIS': 0,
        'ASX': 0, 'GLX': 0, 'UNK': 0
    }

    for i in range(len(amino_info)-1):

        if amino_info[i][0]==amino_info[i+1][0] and not(amino_info[i][1]==amino_info[i+1][1]):
            amino_hist[amino_info[i][0]] += 1
        elif not(amino_info[i][0]==amino_info[i+1][0]):
            amino_hist[amino_info[i][0]] += 1

    return amino_hist


def main():


    contents = ''

    for TARGET_DIR in TARGET_DIRS:

        for pdb_path in natsorted(glob.glob(os.path.join(TARGET_DIR, 'pockets/*.pdb'))):
            # print(pdb_path)

            with open(pdb_path, mode='r') as f:
                amino_info = []
                for line in f:
                    line = line.replace('\n', '')
                    if line.split()[0]=='ATOM':
                        if not (len(line.split()[3])==1 or len(line.split()[3])==5 or len(line.split()[4])==5):
                            amino_info.append([line.split()[3], line.split()[5]])
                        elif not(len(line.split()[3])==5) and len(line.split()[4])==5:
                            amino_info.append([line.split()[3], line.split()[4][1:]])
                        elif len(line.split()[3])==5 and len(line.split()[4])==5:
                            amino_info.append([line.split()[2][-3:], line.split()[4][1:]])
                        elif len(line.split()[3])==5 and not len(line.split()[4])==5:
                            amino_info.append([line.split()[2][-3:], line.split()[5]])
                        else:
                            amino_info.append([line.split()[2][-3:], line.split()[4]])

            amino_info = sorted(amino_info, key=lambda x:x[1])
            
            # print(amino_info)
            
            pdb_name = pdb_path.split('/')[-3][:-4]

            pocket_num = pdb_path.split('/')[-1][:-8].strip('pocket')

            if pdb_name=='1R1WA':
                # print(TARGET_DIR, pocket_num, amino_info)s
                amino_hist = cnt_amino_acid(amino_info)
                print(amino_hist['MET'])
                # sys.exit(1)

            # print(TARGET_DIR, pocket_num, amino_info)
            
            amino_hist = cnt_amino_acid(amino_info)

            # print(amino_hist)

           
            contents = cnt_amino_csv(contents, pdb_name, pocket_num, amino_hist)


    # print(contents)

    
    with open(OUTPUT_PATH, mode='w') as f:
        f.write(contents)

    print('OK.')




if __name__ == '__main__':
    main()