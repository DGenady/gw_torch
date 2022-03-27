import torch
import boto3
from urllib.parse import urlparse
from io import BytesIO
import numpy as np

s3 = boto3.resource('s3',endpoint_url = 'https://s3-west.nrp-nautilus.io')
    
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

print(torch.version.cuda)

            
def loadFile(path, num, s3obj):
    data = np.load(load_to_bytes(s3obj,f's3://tau-astro/gdevit/{path}{num}.npy'), 
                       allow_pickle=True)
    data = torch.from_numpy(data)
    data = torch.permute(data,(0,1,4,2,3))
    return data.float()

def load_to_bytes(s3,s3_uri:str):
    parsed_s3 = urlparse(s3_uri)
    f = BytesIO()
    s3.meta.client.download_fileobj(parsed_s3.netloc, 
                                    parsed_s3.path[1:], f)
    f.seek(0)
    return f

data = loadfile(path='triplet_noise/noise010/file',num=0,s3obj=s3)
if data is not None:
  print('data loaded')
  
