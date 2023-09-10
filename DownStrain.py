import os
import time
import urllib
import argparse
from multiprocessing.pool import ThreadPool


parser = argparse.ArgumentParser(description='PyTorch Triplet network')
parser.add_argument('--start-file', type=int, default=0, metavar='N',
                    help='first file to upload to S3 storage (default: 0)')
parser.add_argument('--last-file', type=int, default=10, metavar='N',
                    help='last file to upload to S3 storage (default: 10)')
args = parser.parse_args()

max_files = 10
project_name = "m4443"
project_path = f"/global/cfs/cdirs/{project_name}/O2_data/"

with open('./gw_torch/spec_prep/O2_H1.txt','r') as f:
    H1_list = f.read().split('\n')

with open('./gw_torch/spec_prep/O2_L1.txt','r') as f:
    L1_list = f.read().split('\n')

H1_save_paths = ['H1_' + x.split('-')[-2] + '.hdf5' for x in H1_list]
L1_save_paths = ['L1_' + x.split('-')[-2] + '.hdf5' for x in L1_list]

H1_save_paths = [os.path.join(project_path, x) for x in H1_save_paths]
L1_save_paths = [os.path.join(project_path, x) for x in L1_save_paths]

with ThreadPool(32) as pool:
    H_combined_list = zip(H1_list, H1_save_paths)[:max_files]
    L_combined_list = zip(L1_list, L1_save_paths)[:max_files]
    combined = H_combined_list + L_combined_list
    done = pool.map(urllib.request.urlretrieve, combined)
print(done)
