import os
import pickle
from signal_generators import *
from spectrogram import make_spectrogram
from file_processing import *
from mpi4py import MPI
import json


def get_chunks_idxs(strain, segment_len):
    chunks = []
    for i in range(0, strain.shape[0]-segment_len-1, segment_len):
        chunk = strain[i:i+segment_len]
        if not np.isnan(chunk).any():
            chunks.append((i,i+segment_len))
    return chunks

def scale_signal_to_snr(chunk, signal, known_snr):
    snr = 10*np.log10(np.linalg.norm(signal)/np.linalg.norm(chunk)).value
    new_sg = signal * np.power(10, (known_snr - snr) / 10 )
    return new_sg

def create_dataset(signal_func, strain, segment_duration, **kwargs):
    print(f"building dataset for {signal_func.__name__}:")
    results = []
    time_offset = 1  # spectrogram takes of one second on each side.
    dt = strain.dt
    segment_len = int((segment_duration / dt).value)
    chunks_idxs = get_chunks_idxs(strain, segment_len)
    for chunk_start, chunk_end in chunks_idxs:
        print("processing chunk")

        signal_start_time = np.random.uniform(-0.3, 1.2)
        snr = np.random.uniform(-44.0, -51.0)
        taper_duration = np.random.uniform(0.01, 0.5)

        chunk = strain[chunk_start:chunk_end]
        t_inj = chunk.t0.value
        signal, signal_params = signal_func(dt=dt.value, times=chunk.times.value, t_inj=t_inj, **kwargs)
        # get t0 alligned with chunk times
        signal.t0 = chunk.t0 + (int(((time_offset + signal_start_time) / chunk.dt).value) * chunk.dt)
        signal = signal.taper(side='leftright', duration=taper_duration)# * scale_factor
        signal = signal + chunk.mean()
        signal = scale_signal_to_snr(chunk, signal, snr)
        combined = chunk.inject(signal)

        # store signal params in dict for dataset
        signal_params['signal_start_time'] = signal_start_time
        signal_params['snr'] = snr
        signal_params['taper_duration'] = taper_duration
        signal_params['signal_duration'] = signal[abs(signal.value)>1e-26].duration.value
        signal_spec,  sig_SNR_offset, sig_SNR_norm, _ = make_spectrogram(combined)
        clean_spec,  clean_SNR_offset, clean_SNR_norm, _ = make_spectrogram(chunk)
        signal_params['sig_SNR_offset'] = sig_SNR_offset
        signal_params['sig_SNR_norm'] = sig_SNR_norm
        signal_params['clean_SNR_offset'] = clean_SNR_offset
        signal_params['clean_SNR_norm'] = clean_SNR_norm
        
        results.append({
            "signal_spec": signal_spec,
            "clean_spec": clean_spec,
            # "signal": signal,
            # "chunk": chunk,
            "signal_params": signal_params
        })
    print(f"done building dataset for {signal_func.__name__}:")

    return results

def process_file(strain_file, detector, save_path):
    print(f"loading file... {strain_file}")
    strain = TimeSeries.read(strain_file, format='hdf5.gwosc', verbose=False)
    print("complete")
    segment_duration = 256
    signal_functions = [
        gw_signal,
        sine_gaussian,
        ringdown,
        chirp_gaussian_inc,
        double_chirp_gaussian,
        square_signal
    ]
    failed = []
    for sig_func in signal_functions:
        try:
            sig_dataset = create_dataset(sig_func, strain, segment_duration, detector=detector)
            pickle_path = f"{save_path}_{sig_func.__name__}.pkl"
            with open(pickle_path, "wb") as f:
                pickle.dump(sig_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            failed.append({"strain_file": strain_file, "sig_func": sig_func.__name__, "e": str(e)})
    return failed

if __name__ == "__main__":
    project_name = 'm4443'
    GWOSC_run = 'O3'
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)
    data_path = f'/global/cfs/cdirs/{project_name}/{GWOSC_run}_data/'
    save_path = f"/global/cfs/cdirs/{project_name}/synth_data/{GWOSC_run}_synth/"

    data_files = os.listdir(data_path)
    H_files = [x for x in data_files if x.startswith("H")]
    L_files = [x for x in data_files if x.startswith("L")]

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
        with open(os.path.join(save_path, f"failed_worker_{rank}.json"), "w") as f:
            json.dump(failed, f)
    print(f"worker {rank}: existing")