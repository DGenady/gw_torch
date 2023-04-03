# getting the strain data saved on S3
from gwpy.timeseries import timeseries
from gwpy.frequencyseries import FrequencySeries
from gwosc.datasets import event_gps
from gwpy.signal.window import recommended_overlap
import scipy.signal as scisig
import numpy as np
import time
import boto3
import os 
import multiprocessing as mp
import random
import pickle

class GWDATA:
    
    def __init__(self, H=None, L=None, times=None):
        self.H = H
        self.L = L
        self.times = times
    
    def save(self,path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

def get_unprocceced_segments(segments,procceced_segments):
    left_to_procces = []
    for i in range(len(segments)):
        done = 0
        for j in range(len(procceced_segments)):
            if segment_list[i][0] == procceced[j][0] and segment_list[i][1] == procceced[j][1]:
                done += 1
        if done == 0:
            left_to_procces.append(i)
    print(f'segments left to be procceced {len(left_to_procces)}. difference {len(segments)-len(procceced)}')
    
    left_segments=[]
    for i in range(len(segments)):
        if i in left_to_procces:
            left_segments.append(segments[i])
    
    return left_segments

def get_segments(path):
    with open(path,'r') as f:
        segment_list = f.read().split('\n')
    done_list = []
    for segment in segment_list:
        if segment == '':
            continue
        t_i = int(segment.split(' ')[0])
        t_f = int(segment.split(' ')[1])
        done_list.append((t_i,t_f))
    return done_list

def segments_files(detector, segment_list=None):
    
    with open(f'./gw_torch/spec_prep/O1_{detector}1.txt','r') as f:
        my_list = f.read().split('\n')

    s3_list = []

    for item in my_list:
        s3_list .append(f'{detector}1_' + item.split('-')[-2] + '.hdf5')

    detector_segment_files = []
    for segment in segment_list:
        t_i = segment[0]
        t_f = segment[1]
        files = []
        for item in s3_list:
            if int(item[3:-5]) <= t_i and int(item[3:-5])+4096 >= t_i:
                files.append(item)
            if int(item[3:-5]) >= t_i and int(item[3:-5]) < t_f:
                files.append(item)
        detector_segment_files.append(files)
    return detector_segment_files
    

def download_segment(segment):
    """
        the input is a list of files needed to be downloaded
    """
    detector = segment[0][:1]
    s3 = boto3.client('s3',endpoint_url='https://s3-west.nrp-nautilus.io')
    
    for file in segment:
        s3_path = f"gdevit/gw_data/strain/O1/{detector}1/{file}"
        save_file = f'{detector}-{detector}1_LOSC_16_V1-{file[3:-5]}-4096.hdf5'
        
        s3.download_file('tau-astro', s3_path,save_file)
        
def delete_segment(segment, detector):
    for file in segment:
        save_file = f'{detector}-{detector}1_LOSC_16_V1-{file[3:-5]}-4096.hdf5'
        os.remove(save_file)
        
def get_chunks(t_i, t_f, Tc=16, To=2):
    
    if t_i + Tc > t_f:
        return [(t_i, t_f)]

    chunks = []
    c_i = t_i
    while (c_i + Tc) <= t_f:
        chunks.append((c_i, c_i + Tc))
        c_i += (Tc - To)
        
    if chunks[-1][1] < t_f:
        chunks.append((t_f - Tc, t_f))
    return chunks

def make_spectrogram(data, Tc=256, To=2, Q=(16,16)):
    
    if np.isnan(data.value).any():
        print('Nan in chunk')
        return
        
    data = data - data.mean()
    data.highpass(frequency=20, filtfilt=True)
    
    Nc = len(data)
    Tc = Nc * data.dt.value
    window = scisig.tukey(M=Nc, alpha=1.0*To/Tc, sym=True)
    data = data * window
    
    window = 'hann'
    fftlength = timeseries._fft_length_default(data.dt)
    overlap = None
    method = "median"
    if overlap is None and fftlength == data.duration.value:
        method = "median"
        overlap = 0
    elif overlap is None:
        overlap = recommended_overlap(window) * fftlength

    ASD = data.asd(fftlength, overlap, window=window, method=method)

    if isinstance(ASD, FrequencySeries):
        with np.errstate(all='raise'):
            whitened_data = data.whiten(asd=ASD, fduration=2,
                               highpass=None)
    
    # desired spectrogram duration and resolution
    duration = 2
    tres = duration/256
    fres = 256
    
    # Q transform sections
    original_size = 4 
    step = 2
    i = 0 

    SNR_offset,SNR_norm, specs, times = [], [], [], []
    
    while i + original_size < whitened_data.duration.value:

        cropped = whitened_data[i*frq:(i+original_size)*frq]
        qt = cropped.q_transform(frange=(10, 2048), qrange=q, whiten=False, tres=tres, fres=fres, logf=True)
        qt = qt[fres//2:-fres//2]
        SNR_offset.append(qt.min())
        qt += np.abs(qt.min())

        SNR_norm.append(qt.max())
        specs.append((qt/qt.max()).value)
        times.append(qt.t0.value)

        i += step

    times = np.array(times)
    SNR_offset = np.array(SNR_offset)
    SNR_norm = np.array(SNR_norm)
    specs = np.array(specs)
    
    return specs, SNR_offset, SNR_norm, times

def get_TS_data(segment_files,t_i ,t_f):
    files = []
    for file in segment_files:
        detector = file[0:1]
        files.append(f'{detector}-{detector}1_LOSC_16_V1-{file[3:-5]}-4096.hdf5')

    data = TimeSeries.read(files, start=t_i, end=t_f, format='hdf5.gwosc',verbose=False)
    return data

def make_save_spec(segment,files):
    
    save_ind = 0

    start_time = time.time()
    
    download_segment(files['H'])
    download_segment(files['L'])
    
    first = True
    
    chunks = get_chunks(t_i=segment[0], t_f=segment[1], Tc=256, To=2)
    
    for chunk in chunks:
        H_strain = get_TS_data(files['H'], chunk[0], chunk[1])
        L_strain = get_TS_data(files['L'], chunk[0], chunk[1])
        if first:
            H_spectrograms, H_SNR_offset, H_SNR_norm, H_times = make_spectrogram(H_strain)
            L_spectrograms, L_SNR_offset, L_SNR_norm, L_times = make_spectrogram(L_strain)
            first = False
        else:
            temp_spec, temp_snr_off, temp_snr_norm, temp_times = make_spectrogram(H_strain)
            H_spectrograms = np.concatenate((H_spectrograms,temp_spec))
            H_SNR_offset = np.concatenate((H_SNR_offset,temp_snr_off))
            H_SNR_norm = np.concatenate((H_SNR_norm,temp_snr_norm))
            H_times = np.concatenate((H_times,temp_times))
            
            temp_spec, temp_snr_off, temp_snr_norm, temp_times = make_spectrogram(L_strain)
            L_spectrograms = np.concatenate((L_spectrograms, temp_spec))
            L_SNR_offset = np.concatenate((L_SNR_offset, temp_snr_off))
            L_SNR_norm = np.concatenate((L_SNR_norm, temp_snr_norm))
            L_times = np.concatenate((L_times,temp_times))
            
        if (H_times != L_times).any() :
            print('time between detector is incorrect')

        if H_spectrograms.shape[0] >= 1000:
            
            file_name = f'segment_{H_times[0]}_{H_times[999]}.gwdata'
            
            gw_data = GWDATA(H = {'spectrograms':H_spectrograms[:1000],
                                   'SNR':{'offset':H_SNR_offset[:1000], 'norm':H_SNR_norm[:1000]}},
                             L = {'spectrograms':L_spectrograms[:1000],
                                   'SNR':{'offset':L_SNR_offset[:1000], 'norm':L_SNR_norm[:1000]}},
                             times = H_times[:1000])
            
            gw_data = pickle.dumps(gw_data)
            s3.put_object(Bucket='tau-astro', Key='gdevit/gw_data/O1/Both/'+file_name, Body=gw_data)
            
            H_spectrograms = H_spectrograms[1000:]
            H_SNR_offset = H_SNR_offset[1000:]
            H_SNR_norm = H_SNR_norm[1000:]
            H_times = H_times[1000:]
            
            L_spectrograms = L_spectrograms[1000:]
            L_SNR_offset = L_SNR_offset[1000:]
            L_SNR_norm = L_SNR_norm[1000:]
            L_times = L_times[1000:]
            
            
            
    delete_segment(files['H'],'H')
    delete_segment(files['L'],'L')
    
    gw_data = GWDATA(H = {'spectrograms':H_spectrograms,
                           'SNR':{'offset':H_SNR_offset, 'norm':H_SNR_norm}},
                     L = {'spectrograms':L_spectrograms,
                           'SNR':{'offset':L_SNR_offset, 'norm':L_SNR_norm}},
                     times = H_times)

    gw_data = pickle.dumps(gw_data)
    file_name = f'segment_{H_times[0]}_{H_times[-1]}.gwdata'
    s3.put_object(Bucket='tau-astro', Key='gdevit/gw_data/O1/Both/'+file_name, Body=gw_data)
    
    print(f'finished segment {segment[0]}-{segment[1]} in {((time.time()-start_time)/60)}f minutes')
    
    with open(f'saved_segments.txt', 'a') as fp:
        fp.write(f'{segment[0]} {segment[1]}\n')
    s3.upload_file('saved_segments.txt', 'tau-astro', f'gdevit/gw_data/O1/Both/saved_segments.txt')
   
    
print(f'Creating spectrograms for both detectors')

s3 = boto3.client('s3',endpoint_url='https://s3-west.nrp-nautilus.io')

segment_list = get_segments('./gw_torch/spec_prep/O1_BOTH.txt')

try:
    s3.download_file('tau-astro', f'gdevit/gw_data/O1/Both/saved_segments.txt','saved_segments.txt')
    procceced = get_segments(f'saved_segments.txt')
    segment_list = get_unprocceced_segments(segment_list, procceced)

except:
    pass

H_files_for_segment = segments_files('H',segment_list)
L_files_for_segment = segments_files('L',segment_list)
indices = list(zip(segment_list, H_files_for_segment, L_files_for_segment))
random.shuffle(indices)

segment_list, H_files_for_segment, L_files_for_segment = zip(*indices)

numCPUs = mp.cpu_count() - 1

if numCPUs > 32:
    pool = mp.Pool(32)
    print(f'using 32 cpus')
else:    
    pool = mp.Pool(mp.cpu_count() - 1)
    print(f'using {mp.cpu_count()-1} cpus')

pool.starmap(make_save_spec, [(segment,{'H':H_files_for_segment[i],'L':L_files_for_segment[i]}) for i, segment in enumerate(segment_list)])
print('done')
