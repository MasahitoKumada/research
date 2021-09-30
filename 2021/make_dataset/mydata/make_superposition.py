import pymol
import os
import glob

base_dir = "/Users/mkumada/Documents/class/2020/work/critipic_pocket/mydata"

# train or test data
data_type = "test"

# for paper critipic binding site
# setting for apo
in_dir, out_dir = "./input/"+data_type+"/apo2", "./output/"+data_type+"/apo2"
# setting for holo
# in_dir, out_dir = "./input/"+data_type+"/holo2", "./output/"+data_type+"/holo2"

# for moad site
# in_dir, out_dir = "./input/train/LBSp_dataset/data_files/holo/input2/", "./input/train/LBSp_dataset/data_files/holo/output/"

pdbid_str_num = 5 # pdbidの長さ 4 or 5

pymol.finish_launching()

for in_file in sorted(glob.glob(os.path.join(in_dir, "*.pdb"))):

    targete_out_dir = in_file.replace(in_dir, out_dir).split('.pdb')[0] + "_out"
    pdb_name = targete_out_dir[-(pdbid_str_num+4):-4]

    pymol.cmd.load(in_file)
    os.chdir(targete_out_dir) 
    pymol.cmd.load(pdb_name + ".pml")
    pymol.cmd.remove("solvent")
    pymol.cmd.zoom()
    pymol.cmd.save(pdb_name + ".pse")
    pymol.cmd.delete("all")

    os.chdir(base_dir)
    # print(os.getcwd())  

    print(pdb_name + " finished.")

pymol.cmd.quit()
print("ok")