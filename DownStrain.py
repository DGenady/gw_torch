import os
import requests
from multiprocessing.pool import ThreadPool

def download_file(url, save_path):
    resp = requests.get(url, allow_redirects=True)
    if resp.status_code != 200:
        return url, False
    with open(save_path, "wb") as f:
        f.write(resp.content)
    print(f"downloading file {save_path}")
    return url, True

max_files = 500
project_name = "m4443"
project_path = f"/global/cfs/cdirs/{project_name}/O2_data/"



with open('./spec_prep/O2_H1.txt','r') as f:
    H1_list = f.read().split('\n')

with open('./spec_prep/O2_L1.txt','r') as f:
    L1_list = f.read().split('\n')

H1_save_paths = ['H1_' + x.split('-')[-2] + '.hdf5' for x in H1_list if len(x)>0]
L1_save_paths = ['L1_' + x.split('-')[-2] + '.hdf5' for x in L1_list if len(x)>0]

H1_save_paths = [os.path.join(project_path, x) for x in H1_save_paths]
L1_save_paths = [os.path.join(project_path, x) for x in L1_save_paths]

with ThreadPool(32) as pool:
    H_combined_list = list(zip(H1_list, H1_save_paths))[:max_files]
    L_combined_list = list(zip(L1_list, L1_save_paths))[:max_files]
    combined = H_combined_list + L_combined_list
    done = pool.starmap(download_file, combined)
print(done)
