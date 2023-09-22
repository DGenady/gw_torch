import os
import pickle
from tqdm import tqdm

project_name = "m4443"
cfs_path = f"/global/cfs/projectdirs/{project_name}/ast_training_data/"
OGWSC_run  = "O1"
data_path = os.path.join(cfs_path, f"{OGWSC_run}_training_data/")
files = [os.path.join(data_path, x) for x in os.listdir(data_path)]
total_H, total_L = 0, 0
for file in tqdm(files):
    with open(file, "rb") as f:
        specs = pickle.load(f)
        assert isinstance(specs, list)
    if len(specs) == 0:
        os.remove(file)
        print(f"removed {file}")
    else:
        if file.split("/")[-1].startswith("H"):
            total_H += len(specs)
        else:
            total_L += len(specs)
with open("./specs_counts.txt", "w") as f:
    f.writelines([
        f"H counts: {total_H}\n",
        f"L counts: {total_L}\n"
        ])
