import os
import pickle
from tqdm import tqdm
from config.conf import conf

project_name = conf['project_config']['project_name']
cfs_path = f"/global/cfs/projectdirs/{project_name}/ast_training_data/"
scratch_path = os.environ['PSCRATCH']
OGWSC_run = conf['run_config']['OGWSC_run']
data_path = os.path.join(cfs_path, f"{OGWSC_run}_training_data/")
output_path = os.path.join(scratch_path, f"{OGWSC_run}_training_data_dist/")
if not os.path.isdir(output_path):
    os.makedirs(output_path, exist_ok=True)

H_files = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.startswith("H")]
L_files = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.startswith("L")]
files_written = 0


H_buffer = []
L_buffer = []
print_every = 100
while H_files and L_files:
    if not H_buffer:
        h_file = H_files.pop(0)
        print(h_file)
        with open(h_file, "rb") as f:
            specs = pickle.load(f)
        if not isinstance(specs, list) or not specs:
            continue
        H_buffer += specs
    if not L_buffer:
        l_file = L_files.pop(0)
        print(l_file)
        with open(l_file, "rb") as f:
            specs = pickle.load(f)
        if not isinstance(specs, list) or not specs:
            continue
        L_buffer += specs
    output_file_name = f"specs-out-{len(H_files)}-{len(H_buffer)}-{len(L_files)}-{len(L_buffer)}.pkl"
    with open(os.path.join(output_path, output_file_name), "wb") as f:
        pickle.dump({"H": H_buffer.pop(0), "L": L_buffer.pop(0)}, f)
        files_written += 1
    if files_written % print_every == 0:
        print(f"files_writen: {files_written}, remaining - H: {len(H_files)} L: {len(L_files)}")

with open("./specs_counts.txt", "w") as f:
    f.writelines([
        f"files written to {output_path}: {files_written}"
        ])
