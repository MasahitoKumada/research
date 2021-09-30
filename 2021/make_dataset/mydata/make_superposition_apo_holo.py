import pymol
import os
import glob


# train or test data
data_type = "test"

# for paper critipic binding site
dir_holo, dir_apo = "./input/"+data_type+"/holo2", "./output/"+data_type+"/apo2"

in_apo_file = "./output/"+data_type+"/"+data_type+"_apo_list.txt"
in_holo_file = "./output/"+data_type+"/"+data_type+"_holo_list.txt"


in_apo_filename_lst = None
in_holo_filename_lst = None
with open(in_apo_file) as f:
    in_apo_filename_lst = f.readlines()[1:]
with open(in_holo_file) as f:
    in_holo_filename_lst = f.readlines()[1:]


# pymol.finish_launching()

for apo_file, holo_file in zip(in_apo_filename_lst, in_holo_filename_lst):

    apo_file = apo_file.rstrip('\n')
    holo_file = holo_file.rstrip('\n')

    # apoのfpocketの結果にholoのpdbを重ね合わせる.
    apo_filedir = os.path.join(dir_apo, apo_file + '_out')
    apo_filepath = os.path.join(apo_filedir, apo_file + '.pse')
    holo_filepath = os.path.join(dir_holo, holo_file + '.pdb')

    # pymol draw
    pymol.cmd.load(apo_filepath)
    pymol.cmd.load(holo_filepath)
    pymol.cmd.align(mobile=holo_file, target=apo_file)
    pymol.cmd.remove("solvent")
    pymol.cmd.zoom()
    pymol.cmd.save(os.path.join(apo_filedir, apo_file) + "_holo.pse")
    pymol.cmd.delete("all")

    print(apo_file + " finished.")
    # print(os.getcwd()) 

pymol.cmd.quit()
print("ok")