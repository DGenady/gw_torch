import numpy as np
import pickle
import os
import boto3
import random
import multiprocessing as mp

class GWDATA:
    
    def __init__(self, H=None, L=None, times=None):
        self.H = H
        self.L = L
        self.times = times
     
        
    def save(self,path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_shuffle(save_ind,files=[]):
    
    first = True
    
    for file in files:
        
        s3 = boto3.client('s3', endpoint_url='https://s3-west.nrp-nautilus.io')

        save_name = file.split('/')[-1]
        s3.download_file('tau-astro', file, save_name)

        with open(save_name, 'rb') as f:
            td = pickle.load(f)

        if first:
            H_spec, H_offset, H_norm = td.H['spectrograms'], td.H['SNR']['offset'], td.H['SNR']['norm']
            L_spec, L_offset, L_norm = td.L['spectrograms'], td.L['SNR']['offset'], td.L['SNR']['norm']
            times = td.times
            first = False
        else:
            H_spec = np.vstack((H_spec,td.H['spectrograms']))
            H_offset = np.hstack((H_offset,td.H['SNR']['offset']))
            H_norm = np.hstack((H_norm,td.H['SNR']['norm']))

            L_spec = np.vstack((L_spec,td.L['spectrograms']))
            L_offset = np.hstack((L_offset,td.L['SNR']['offset']))
            L_norm = np.hstack((L_norm,td.L['SNR']['norm']))

            times = np.hstack((times,td.times))
                
        os.remove(save_name)
        
    inds = np.arange(H_spec.shape[0])
    np.random.shuffle(inds)
    
    H_spec, H_offset, H_norm = H_spec[inds], H_offset[inds], H_norm[inds]
    L_spec, L_offset, L_norm = L_spec[inds], L_offset[inds], L_norm[inds]
    times = times[inds]
    
    while H_spec.shape[0] > 1000:
        
        to_up = GWDATA(H={'spectrograms':H_spec[:1000], 'SNR':{'offset':H_offset[:1000],'norm':H_norm[:1000]}},
                      L={'spectrograms':L_spec[:1000], 'SNR':{'offset':L_offset[:1000],'norm':L_norm[:1000]}}, 
                      times = times[:1000])
        
        to_up = pickle.dumps(to_up)
        file_name = f'data_{save_ind}.gwdata'
        s3.put_object(Bucket='tau-astro', Key='gdevit/gw_data/O1/shuffled/'+file_name, Body=to_up)
        save_ind += 1
        
        H_spec, H_offset, H_norm = H_spec[1000:], H_offset[1000:], H_norm[1000:]
        L_spec, L_offset, L_norm = L_spec[1000:], L_offset[1000:], L_norm[1000:]
        times = times[1000:]
        
    to_up = GWDATA(H={'spectrograms':H_spec, 'SNR':{'offset':H_offset,'norm':H_norm}},
                      L={'spectrograms':L_spec, 'SNR':{'offset':L_offset,'norm':L_norm}}, 
                      times = times)
        
    to_up = pickle.dumps(to_up)
    file_name = f'data_{save_ind}.gwdata'
    s3.put_object(Bucket='tau-astro', Key='gdevit/gw_data/O1/shuffled/'+file_name, Body=to_up)

    
    
s3 = boto3.resource('s3',endpoint_url='https://s3-west.nrp-nautilus.io')
my_bucket = s3.Bucket('tau-astro')

O1_files = []

for object_summary in my_bucket.objects.filter(Prefix="gdevit/gw_data/O1/Both/"):
    O1_files.append(object_summary.key)
    
O1_files.remove('gdevit/gw_data/O1/Both/saved_segments.txt')
    
random.shuffle(O1_files)

numCPUs = mp.cpu_count() - 1

if numCPUs > 32:
    pool = mp.Pool(32)
    print(f'using 32 cpus')
else:    
    pool = mp.Pool(mp.cpu_count() - 1)
    print(f'using {mp.cpu_count()-1} cpus')
    
fpt = 10 # files per thread

pool.starmap(save_shuffle, [(i*fpt, O1_files[i*fpt:(i+1)*fpt]) for i in range(len(O1_files[:50])//fpt)])
print('done')
