from gwpy.timeseries import TimeSeries
import scipy.signal as scisig
import numpy as np
import time
import boto3
import os 
import multiprocessing as mp
import random

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
    
    with open(f'./gw_torch/spec_prep/O2_{detector}1.txt','r') as f:
        my_list = f.read().split('\n')

    detector_segment_files = []
    for segment in segment_list:
        t_i = segment[0]
        t_f = segment[1]
        files = []
        for item in my_list:
            if item == '':
                continue
            t0 = int(item.split('-')[-2])
            if t0 <= t_i and t0+4096 >= t_i:
                files.append(item.split('/')[-1])
            if t0 >= t_i and t0 < t_f:
                files.append(item.split('/')[-1])
        detector_segment_files.append(files)
    
    return detector_segment_files
    

def download_segment(segment,detector):
    """
        the input is a list of files needed to be downloaded
    """
    s3 = boto3.client('s3',endpoint_url='https://s3-west.nrp-nautilus.io')
    
    for file in segment:
        s3_path = f"gdevit/gw_data/strain/O2/{detector}1/{file}"
        s3.download_file('tau-astro', s3_path, file)
        
def delete_segment(segment,detector):
    for file in segment:
        os.remove(file)
        
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

def make_spectrogram(data, Tc=64, To=2):
    
    if np.isnan(data.value).any():
        print('Nan in chunk')
        return
        
    data = data - data.mean()
    data.highpass(frequency=20, filtfilt=True)

    Nc = len(data)
    Tc = Nc * data.dt.value
    window = scisig.tukey(M=Nc, alpha=1.0*To/Tc, sym=True)
    data = data * window

    duration = 2
    tres = duration/256
    fres = 256
    
    qt = data.q_transform(frange=(10,2048), qrange=(4,64), whiten=True, tres=tres, fres=fres, logf=True)
    qt = qt[int(To/duration / qt.dt.value):-int(To/duration / qt.dt.value)]
    
    two_sec_slice = int(2/tres)
    num_of_slices = qt.shape[0]//two_sec_slice
    
    qts = np.empty((num_of_slices,256,256))
    times = np.empty(num_of_slices)
    for i in range(num_of_slices):
        qts[i] = qt[i*two_sec_slice:(i+1)*two_sec_slice].value
        times[i] = qt[i*two_sec_slice:(i+1)*two_sec_slice].t0.value
        qts[i] = qts[i]+np.abs(qts[i].min())
        qts[i] = qts[i]/qts[i].max()
        
    return qts, times

def get_TS_data(segment_files,t_i ,t_f):
    files = []
    for file in segment_files:
        detector = file[0:1]
        files.append(f'{detector}-{detector}1_LOSC_16_V1-{file[3:-5]}-4096.hdf5')

    data = TimeSeries.read(files, start=t_i, end=t_f, format='hdf5.gwosc',verbose=False)
    return data

def make_save_spec(segment,files,detector):
    
    save_ind = 0

    start_time = time.time()
    
    download_segment(files)
    
    first = True
    
    chunks = get_chunks(t_i=segment[0], t_f=segment[1], Tc=64, To=2)
    
    for chunk in chunks:
        strain = get_TS_data(files, chunk[0], chunk[1])
        if first:
            spectrograms, times = make_spectrogram(strain)
            first = False
        else:
            temp_spec, temp_times = make_spectrogram(strain)
            spectrograms = np.concatenate((spectrograms,temp_spec))
            times = np.concatenate((times,temp_times))

        if spectrograms.shape[0] >= 1000:
            file_name = f'{detector}_segment_{segment[0]}_{segment[1]}_{save_ind}.npy'
            time_name = f'{detector}_segment_times_{segment[0]}_{segment[1]}_{save_ind}.npy'
            np.save(file_name,spectrograms[:1000],allow_pickle=True)
            np.save(time_name,times[:1000],allow_pickle=True)
            s3.upload_file(file_name, 'tau-astro', f'gdevit/gw_data/O2/{detector}1/spectrograms/{file_name}')
            s3.upload_file(time_name, 'tau-astro', f'gdevit/gw_data/O2/{detector}1/times/{time_name}')
            os.remove(file_name)
            os.remove(time_name)
            spectrograms = spectrograms[1000:]
            times = times[1000:]
            save_ind += 1
            
    delete_segment(files,detector)
    
    file_name = f'{detector}_segment_{segment[0]}_{segment[1]}_{save_ind}.npy'
    time_name = f'{detector}_segment_times_{segment[0]}_{segment[1]}_{save_ind}.npy'
    np.save(file_name,spectrograms,allow_pickle=True)
    np.save(time_name,times,allow_pickle=True)
    s3.upload_file(file_name, 'tau-astro', f'gdevit/gw_data/O2/{detector}1/spectrograms/{file_name}')
    s3.upload_file(time_name, 'tau-astro', f'gdevit/gw_data/O2/{detector}1/times/{time_name}')
    os.remove(file_name)
    os.remove(time_name)
    print(f'finished segment {segment[0]}-{segment[1]} in {((time.time()-start_time)/60)}f minutes')
    print(f'last saved files is number {save_ind}')
    
    with open(f'saved_segments.txt', 'a') as fp:
        fp.write(f'{segment[0]} {segment[1]}\n')
    s3.upload_file('saved_segments.txt', 'tau-astro', f'gdevit/gw_data/O2/{detector}1/saved_segments.txt')
   
    
detector = 'H'
print(f'Creating spectrograms for {detector} detector')

s3 = boto3.client('s3',endpoint_url='https://s3-west.nrp-nautilus.io')

procceced = None
try:
    s3.download_file('tau-astro', f'gdevit/gw_data/O2/{detector}1/saved_segments.txt','saved_segments.txt')
    procceced = get_segments(f'saved_segments.txt')
except:
    print('no saved segements')
    
    

segment_list = get_segments('./gw_torch/spec_prep/O2_BOTH.txt')

if procceced != None:
    segment_list = get_unprocceced_segments(segment_list, procceced)
    
files_for_segment = segments_files(detector,segment_list)  
indices = list(zip(segment_list, files_for_segment))
random.shuffle(indices)

segment_list, files_for_segment = zip(*indices)

numCPUs = mp.cpu_count() - 1

if numCPUs > 16:
    pool = mp.Pool(16)
    print(f'using 16 cpus')
else:    
    pool = mp.Pool(mp.cpu_count() - 1)
    print(f'using {mp.cpu_count()-1} cpus')

pool.starmap(make_save_spec, [(segment,files_for_segment[i], detector) for i, segment in enumerate(segment_list)])
print('done')
