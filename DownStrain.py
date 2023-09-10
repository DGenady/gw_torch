import os
import requests
from multiprocessing.pool import ThreadPool
from time import perf_counter
from random import shuffle


t1 = perf_counter()
def download_file(url, save_path):
    resp = requests.get(url, allow_redirects=True)
    if resp.status_code != 200:
        return url, False
    with open(save_path, "wb") as f:
        f.write(resp.content)
    print(f"allpsed: {perf_counter() - t1}: downloaded file {save_path}")

    return url, True

max_files = 1000
WGOSC_run = "O3"
project_name = "m4443"
project_path = f"/global/cfs/cdirs/{project_name}/O3_data/"


with open(f'./spec_prep/{WGOSC_run}_H1.txt', 'r') as f:
    H1_list = f.read().split('\n')

with open(f'./spec_prep/{WGOSC_run}_L1.txt','r') as f:
    L1_list = f.read().split('\n')

H1_save_paths = ['H1_' + x.split('-')[-2] + '.hdf5' for x in H1_list if len(x)>0]
L1_save_paths = ['L1_' + x.split('-')[-2] + '.hdf5' for x in L1_list if len(x)>0]

H1_save_paths = [os.path.join(project_path, x) for x in H1_save_paths]
L1_save_paths = [os.path.join(project_path, x) for x in L1_save_paths]
with ThreadPool(16) as pool:
    H_combined_list = list(zip(H1_list, H1_save_paths))[:max_files]
    L_combined_list = list(zip(L1_list, L1_save_paths))[:max_files]
    combined = H_combined_list + L_combined_list
    shuffle(combined)
    done = pool.starmap(download_file, combined)
print(done)
