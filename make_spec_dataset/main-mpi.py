import os
print(os.getcwd())
import sys
sys.path.append('/global/homes/g/gdevit/gw_torch')
import pickle
import numpy as np
from gwpy.timeseries import TimeSeries
from spectrogram import make_spectrogram
from file_processing import *
from mpi4py import MPI
import json

from config.conf import conf
data_conf = conf['data_config']


def get_chunks_idxs(strain, segment_len):
    chunks = []
    for i in range(0, strain.shape[0]-segment_len-1, segment_len):
        chunk = strain[i:i+segment_len]
        if not np.isnan(chunk).any():
            chunks.append((i,i+segment_len))
    return chunks


def create_dataset(strain, segment_duration, **kwargs):
    results = []
    dt = strain.dt
    segment_len = int((segment_duration / dt).value)
    chunks_idxs = get_chunks_idxs(strain, segment_len)
    for chunk_start, chunk_end in chunks_idxs:
        chunk = strain[chunk_start:chunk_end]
        clean_specs, times = make_spectrogram(chunk)

        results += [{"clean_spec": c, "times": t} for c, t in zip(clean_specs, times)]

    return results


def process_file(strain_file, detector, save_path):
    print(f"loading file... {strain_file}")
    strain = TimeSeries.read(strain_file, format='hdf5.gwosc', verbose=False)
    print("complete")
    segment_duration = 256
    failed = []
    try:
        strain_dataset = create_dataset(strain, segment_duration, detector=detector)
        if len(strain_dataset) == 0:
            failed.append({"strain_file": strain_file, "e": "zero specs"})
        pickle_path = f"{save_path}-specs.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(strain_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        failed.append({"strain_file": strain_file, "e": str(e)})
    return failed

if __name__ == "__main__":
    project_name = conf['project_config']['project_name']
    GWOSC_run = conf['run_config']['OGWSC_run']
    max_files = 2000
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    data_path = f'/global/cfs/cdirs/{project_name}/{GWOSC_run}_data/'
    save_path = f"/global/cfs/cdirs/{project_name}/ast_training_data/{GWOSC_run}_training_data/"
    logs_path = os.path.join(save_path, "worker_logs")
    if rank == 0:
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
        if not os.path.isdir(logs_path):
            os.makedirs(logs_path, exist_ok=True)

    data_files = os.listdir(data_path)
    H_files = [x for x in data_files if x.startswith("H")][:max_files]
    L_files = [x for x in data_files if x.startswith("L")][:max_files]

    H_range = int((rank / size) * len(H_files)), int(((rank + 1) / size) * len(H_files))
    L_range = int((rank / size) * len(L_files)), int(((rank + 1) / size) * len(L_files))

    my_H_files = H_files[H_range[0]:H_range[1]]
    my_L_files = L_files[L_range[0]:L_range[1]]
    failed = []
    for file_ind, (H_file, L_file) in enumerate(zip(my_H_files, my_L_files)):
        print(f'working of file number {file_ind} / {min(len(my_L_files), len(my_H_files))}')
        H_file_save = os.path.join(save_path, H_file.split(".")[0])
        H_failed = process_file(os.path.join(data_path, H_file), "H1", H_file_save)

        L_file_save = os.path.join(save_path, L_file.split(".")[0])
        L_failed = process_file(os.path.join(data_path, L_file), "L1", L_file_save)
        failed += H_failed
        failed += L_failed
    if len(failed) > 0:
        with open(os.path.join(logs_path, f"failed_worker_{rank}.json"), "w") as f:
            json.dump(failed, f)
    print(f"worker {rank}: existing")
