import pymol
import os
import glob


## setting for apo
# in_dir = "./apo"
# out_dir = "./apo2"
# in_file = "test_apo_list.txt"

## setting for holo 
in_dir = "./holo"
out_dir = "./holo2"
in_file = "test_holo_list.txt"


in_filename_lst = None

with open(in_file) as f:
    in_filename_lst=sorted(f.readlines()[1:])

pymol.finish_launching()

for file_in_chain in in_filename_lst:

    # print(file_in_chain, in_file)
    target_pdb = file_in_chain.rstrip('\n')[:-1]
    target_chain = file_in_chain.rstrip('\n')[-1]

    in_file = os.path.join(in_dir, target_pdb + '.pdb') 
    out_filename =  os.path.join(out_dir, file_in_chain.rstrip('\n') + '.pdb') 
    # print(in_file, out_filename)
    
    pymol.cmd.load(in_file)
    pymol.cmd.remove("not chain "+target_chain)
    pymol.cmd.save(out_filename)

    pymol.cmd.delete("all")

    print(file_in_chain.rstrip('\n')+" finished.")

pymol.cmd.quit()

print("ok")