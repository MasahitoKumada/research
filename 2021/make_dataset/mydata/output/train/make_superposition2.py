import pymol
import os
import glob

base_dir = "/Users/mkumada/Documents/class/2020/work/critipic_pocket/mydata"

# for paper critipic binding site
dir_holo, dir_apo = "input/train/holo2", "output/train/apo2"

in_apo_file = "./output/train/train_apo_list.txt"
in_holo_file = "./output/train/train_holo_list.txt"


in_apo_filename_lst = None
in_holo_filename_lst = None
with open(in_apo_file) as f:
    in_apo_filename_lst = f.readlines()[1:]
with open(in_holo_file) as f:
    in_holo_filename_lst = f.readlines()[1:]


# pymol.finish_launching()

for apo_file, holo_file in zip(in_apo_filename_lst, in_holo_filename_lst):

    apo_filepath = os.path.join(os.path.join(base_dir, dir_apo), apo_file.rstrip('\n') + '.pdb')
    holo_filepath = os.path.join(os.path.join(base_dir, dir_holo), holo_file.rstrip('\n') + '.pdb')

    print(apo_filepath)
    print(holo_filepath)

    pymol.cmd.load(apo_filepath)
    # os.chdir(targete_out_dir)
    print(os.getcwd())  
    # pymol.cmd.load(pdb_name + ".pml")
    # pymol.cmd.save(pdb_name + ".pse")

    # pymol.cmd.delete("all")

    # os.chdir(base_dir)
    # # print(os.getcwd())  

    # print(pdb_name + " finished.")

# pymol.cmd.quit()
print("ok")