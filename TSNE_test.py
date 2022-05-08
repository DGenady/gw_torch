import numpy as np
from sklearn.manifold import TSNE
import boto3
import torch
import argparse
from tools import *
import time

parser = argparse.ArgumentParser(description='tsne of data')
parser.add_argument('--file-name', default='Embed', type=str,
                    help='File to be embedded by t-SNE')

args = parser.parse_args()

start_time = time.perf_counter()

s3 = boto3.resource('s3',endpoint_url = 'https://s3-west.nrp-nautilus.io')
file_name = args.file_name

data = torch.load(load_to_bytes(s3,f's3://tau-astro/gdevit/model/embedded/{file_name}.pt')).numpy()
samples = data.shape[1]
raw_data = np.concatenate((data[0,:,0,:],data[0,:,2,:]))
signal_data = np.concatenate((data[1,:,0,:],data[1,:,2,:]))
combined_data = np.concatenate((raw_data,signal_data))
embedded = TSNE(n_components=2, learning_rate='auto', init='random',perplexity=100).fit_transform(combined_data)

group = np.zeros(4*samples)
group[1*samples:2*samples] = 1
group[2*samples:3*samples] = 2
group[3*samples:] = 3
                                
np.save(f'{file_name}_embedded.npy', embedded, allow_pickle=True)
np.save(f'{file_name}_group.npy', group, allow_pickle=True)
print('Done in {} hours'.format((time.perf_counter()-start_time)/3600))
                                
