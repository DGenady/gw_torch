import time
import boto3
import os
import pickle
import numpy as np

class GWDATA:
    
    def __init__(self, H, L, times):
        self.H = H
        self.L = L
        self.times = times
      
        
    def save(self,path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            
s3 = boto3.resource('s3',endpoint_url='https://s3-west.nrp-nautilus.io')
my_bucket = s3.Bucket('tau-astro')

O1_files = []

for object_summary in my_bucket.objects.filter(Prefix="gdevit/gw_data/O1/shuffled/"):
    O1_files.append(object_summary.key)

s3 = boto3.client('s3',endpoint_url='https://s3-west.nrp-nautilus.io')

first = True

t0 = time.time()

for file in O1_files:
    save_name = file.split('/')[-1]
    s3.download_file('tau-astro',file,save_name)
    with open(save_name,'rb') as f:
        if first:
            times = pickle.load(f).times
            
        else:
            times = np.hstack((times,pickle.load(f).times))
            
    os.remove(save_name)
    if first:
        print(f'it will take {len(O1_files)*(time.time()-t0)/60}')
        first = False
    
times = times.tolist()
print( len(set(times)) == len(times))
print( len(set(times))/len(times) ) 
