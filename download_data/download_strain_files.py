import os
import requests
from multiprocessing.pool import ThreadPool
from random import shuffle

def download_file(url, save_path):
    resp = requests.get(url, allow_redirects=True)
    if resp.status_code != 200:
        return url, False
    with open(save_path, "wb") as f:
        f.write(resp.content)
    return url, True

if __name__ == '__main__':
    # copy settings from get_gwosc_file_list.py
    dataset = 'o3a_16khz_r1'
    gpsstart = 1238166018
    gpsend = 1253977218
    H_detector_file_list = f'./{dataset}_h1_{gpsstart}_{gpsend}_files.txt'
    L_detector_file_list = f'./{dataset}_l1_{gpsstart}_{gpsend}_files.txt'

    max_files = 2000 # int | None
    num_threads = 64

    WGOSC_run = "O3"
    project_name = "m4443"
    project_path = f"/global/cfs/cdirs/{project_name}/{WGOSC_run}_data/"


    with open(H_detector_file_list, 'r') as f:
        H1_list = f.read().split('\n')

    with open(L_detector_file_list,'r') as f:
        L1_list = f.read().split('\n')

    H1_save_paths = ['H1_' + x.split('-')[-2] + '.hdf5' for x in H1_list if len(x) > 0]
    L1_save_paths = ['L1_' + x.split('-')[-2] + '.hdf5' for x in L1_list if len(x) > 0]

    H1_save_paths = [os.path.join(project_path, x) for x in H1_save_paths]
    L1_save_paths = [os.path.join(project_path, x) for x in L1_save_paths]
    with ThreadPool(num_threads) as pool:
        if max_files is not None:
            H_combined_list = list(zip(H1_list, H1_save_paths))[:max_files]
            L_combined_list = list(zip(L1_list, L1_save_paths))[:max_files]
        else:
            H_combined_list = list(zip(H1_list, H1_save_paths))
            L_combined_list = list(zip(L1_list, L1_save_paths))

        combined = H_combined_list + L_combined_list
        shuffle(combined)
        done = pool.starmap(download_file, combined)
    print(done)
