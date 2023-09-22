import matplotlib.pyplot as plt
import seaborn as sns
from gwpy.timeseries import timeseries
from gwpy.frequencyseries import FrequencySeries
from gwosc.datasets import event_gps
from gwpy.signal.window import recommended_overlap
import boto3
import scipy.signal as scisig
import numpy as np
import random
import pickle
from gwpy.timeseries import TimeSeries
import os
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.detector import Detector
from tqdm import tqdm
from scipy.signal import square
import pickle
from signal_generators import *
from spectrogram import make_spectrogram
from file_processing import *
from multiprocessing import Pool, cpu_count
from functools import partial

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

def process_file(strain_file, detector):
    print(f"loading file... {strain_file}")
    strain = TimeSeries.read(strain_file, format='hdf5.gwosc', verbose=False)
    print("complete")
    segment_duration = 256
    signal_functions = [
        gw_signal,
        sine_gaussian,
        ringdown,
        chirp_gaussian_inc,
        # chirp_gaussian,
        double_chirp_gaussian,
        square_signal
    ]
    pickle_files = []
    f = partial(create_dataset, strain=strain, segment_duration=segment_duration, detector=detector)
    print("starting pool")
    with Pool(cpu_count) as pool:
        print("processing...")
        datasets = pool.map(f, signal_functions)
    for sig_func, sig_dataset in tqdm(zip(signal_functions, datasets)):
        print(f"writing dataset for {sig_func.__name__}:")
        pickle_path = f"./results/{sig_func.__name__}.pkl" 
        with open(pickle_path, "wb") as f:
            pickle.dump(sig_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle_files.append(pickle_path)
    return pickle_files

if __name__ == "__main__":
    cpu_count = min(cpu_count(), 6)
    print(f"using {cpu_count} cores")
    s3 = boto3.client('s3', endpoint_url = 'https://s3-west.nrp-nautilus.io') 
    data_path = r'/home/jovyan/create_spectrogram/'
    save_path = r"tomerh/synth_data_256_v5/"

    segment_list = get_segments(os.path.join(data_path, 'O1_BOTH.txt'))
    H_files_for_segment = segments_files('H',segment_list, data_path=data_path)
    L_files_for_segment = segments_files('L',segment_list, data_path=data_path)
    H_segment_files = set([item for sublist in H_files_for_segment for item in sublist])
    L_segment_files = set([item for sublist in L_files_for_segment for item in sublist])
    
    
    already_done = s3.list_objects_v2(Bucket='tau-astro', Prefix='tomerh/synth_data_256_v5')
    already_done = [x['Key'].split("/")[-1] for x in already_done['Contents']]
    already_done = set([x.split("4096_")[0]+"4096.hdf5" for x in already_done])

    # chunk number 416 was the largest duration ofcontinues measurment

    # file_ind = 416
    # print(segment_list[file_ind][1]-segment_list[file_ind][0])
    # final_spec_num = 1000
    # save_ind = 0

#     duration = 2

#     first = True

    # print(f'number of files in the segment {len(H_files_for_segment[file_ind])}')
    # print(f'estimation for the number of files {97084/4096}')
    # for file_ind in range(400, 450):
    for file_ind, (H_file, L_file) in enumerate(zip(H_segment_files, L_segment_files)):
            print(f'working of file number {file_ind} / {min(len(H_segment_files), len(L_segment_files))}')
            if H_file not in already_done:
                

                H_path = f"gdevit/gw_data/strain/O1/H1/{H_file}"
                H_save_file = f'H-H1_LOSC_16_V1-{H_file[3:-5]}-4096.hdf5'
                s3.download_file('tau-astro',H_path,H_save_file)
                print(f"downloaded file {H_file}")



                H_pickle_files = process_file(H_save_file, "H1")
                print(f"processed H file")
                print(f"total: {len(H_pickle_files)} files")
                for pkl_file in H_pickle_files:
                    pkl_save_path = os.path.join(save_path, f"H-H1_LOSC_16_V1-{H_file[3:-5]}-4096_{pkl_file.split('/')[-1]}")
                    try:
                        s3.upload_file(pkl_file, "tau-astro", pkl_save_path)
                    except Exception as e:
                        print(e)

                    os.unlink(pkl_file)
                os.unlink(H_save_file)
                print(f"uploaded H file")
            else:
                print("already done Hfile")
            if L_file not in already_done:    
                L_path = f"gdevit/gw_data/strain/O1/L1/{L_file}"
                L_save_file = f'L-L1_LOSC_16_V1-{L_file[3:-5]}-4096.hdf5'
                s3.download_file('tau-astro',L_path,L_save_file)
                print(f"downloaded file {L_file}")

                L_pickle_files = process_file(L_save_file, "H1")
                print(f"processed L file")
                print(f"total: {len(L_pickle_files)} files")
                for pkl_file in L_pickle_files:
                    pkl_save_path = os.path.join(save_path, f"L-L1_LOSC_16_V1-{L_file[3:-5]}-4096_{pkl_file.split('/')[-1]}")
                    try:
                        s3.upload_file(pkl_file, "tau-astro", pkl_save_path)
                    except Exception as e:
                        print(e)
                    os.unlink(pkl_file) 
                os.unlink(L_save_file)    
                print(f"uploaded L file")
            else:
                print("already done Lfile")




