

"""
common functions used in my scripts

"""

from urllib.parse import urlparse
from io import BytesIO
import numpy as np
import torch
from scipy.signal import chirp
import numpy as np
from gwpy.timeseries import TimeSeries
from skimage.transform import resize
import boto3
import matplotlib.pyplot as plt


def loadFile(path, num, s3obj):
    """ loads files from s3 server """
    data = np.load(load_to_bytes(s3obj,f's3://tau-astro/gdevit/{path}{num}.npy'), 
                       allow_pickle=True)
    data = torch.from_numpy(data)
    data = torch.permute(data,(0,1,4,2,3))
    data = data.float()
    return data/255

def load_to_bytes(s3,s3_uri:str):
    parsed_s3 = urlparse(s3_uri)
    f = BytesIO()
    s3.meta.client.download_fileobj(parsed_s3.netloc, 
                                    parsed_s3.path[1:], f)
    f.seek(0)
    return f


def getBatchDataClass(data,batchNum,batchsize):
    """ returns a batch of the data to be trained """
    h1 = data[batchNum*batchsize:(batchNum+1)*batchsize,0,:,:,:]
    l1 = data[batchNum*batchsize:(batchNum+1)*batchsize,2,:,:,:]
    
    h1l = torch.zeros(batchsize)
    l1l = torch.zeros(batchsize) + 1
    
    labels = torch.cat((h1l,l1l))
    img = torch.cat((h1,l1))
    
    perm = torch.randperm(img.size(0))
    
    img = img[perm,:,:,:]
    labels = labels[perm]
    return img, labels.type(torch.LongTensor)


def create_chirp():
    
    t = np.linspace(0, 1, int(1*16384))
    param = np.random.randint(10,100,4)
    a1 = np.random.randint(2,5)
    a2 = np.random.randint(8,20)
    f_t = param[0]+param[1]*t+param[2]*np.exp((t)**a1)+param[3]*np.exp((t)**a2)
    sig = (1+np.exp(t))*np.sin(t*f_t)

    chirps = sig
    gauss = np.exp(-(t - 1) ** 4 / 0.43** 2)
    
    total_length = np.random.randint(2,15)
    start_ind = np.random.randint(total_length-1)*16384
    
    mySignal = 0.3*np.random.rand(total_length*16384) - 0.5
    mySignal[start_ind:start_ind+16384] = mySignal[start_ind:start_ind+16384] + chirps*gauss

    series = TimeSeries(mySignal,t0=100, dt=1/16384)
    trans = series.q_transform(whiten=True,frange=(10, 2048), qrange=(4, 64), tres=0.002 )

    plot = trans.plot(figsize=(10, 10),vmin=0, vmax=1e5)
    ax = plot.gca()
    ax.grid(False)
    ax.set_yscale('log')
    
    
    
    plot.subplots_adjust(bottom = 0)
    plot.subplots_adjust(top = 1)
    plot.subplots_adjust(right = 1)
    plot.subplots_adjust(left = 0)
    
    ax.set_axis_off()
    
    extent = ax.get_window_extent().transformed(plot.dpi_scale_trans.inverted())
    extent = ax.get_window_extent().transformed(plot.dpi_scale_trans.inverted())

    plot.canvas.draw()
    values = np.frombuffer(plot.canvas.tostring_rgb(), dtype=np.uint8)
    values = values.reshape(plot.canvas.get_width_height()[::-1] + (3,))
    values = np.flipud(values)
    values = resize(values,(224,224), mode='reflect')
    values = values[::-1,:,:]
    plot.close()
    
    return values


def add_signal(original):
    """
        Provided with data of shape [sample,label,height,width,color] retuns data with additonal 
        random chrip like signals to all 3 different labes.
    """
    samples = original.shape[0]
    for sample in range(samples):
        noise = create_chirp()
        for channel in [0,1,2]:
            original[sample,channel,:,:,:] = np.maximum(original[sample,channel,:,:,:], noise)
    plt.close('all')
    return original

def getBatchData(data,batchNum,batchsize,k):
    imgs = data[batchNum*batchsize:(batchNum+1)*batchsize,k,:,:,:]
    return imgs

def loadFileMod(path, num, s3obj):
    """ loads files from s3 server and adds signal to the data"""
    data = np.load(load_to_bytes(s3obj,f's3://tau-astro/gdevit/{path}{num}.npy'), 
                       allow_pickle=True)
    data = torch.from_numpy(data)
    data = torch.permute(data,(0,1,4,2,3))
    data = data.float()
    return data/255

def data_to_torch(data):
    data = torch.from_numpy(data)
    data = torch.permute(data,(0,1,4,2,3))
    data = data.float()
    return data

def add_signal_class(original,noise):
    """
        Provided with data of shape [sample,label,height,width,color] retuns data with additonal 
        random chrip like signals to all 3 different labes.
    """
    indicies = np.arange(1000)
    np.random.shuffle(indicies)
    samples = original.shape[0]
    for sample in range(samples):
        for channel in [0,1,2]:
            index = indicies[sample]
            original[sample,channel,:,:,:] = np.maximum(original[sample,channel,:,:,:],noise[index,:,:,:])
    return original
