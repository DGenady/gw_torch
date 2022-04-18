import torch
import torch.nn as nn
import torch.nn.functional as F
import boto3
from urllib.parse import urlparse
from io import BytesIO
import numpy as np
from torchvision import models
import time
from resnetNoBN import resnet50NoBN
import argparse


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Triplet network')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-7, metavar='LR',
                    help='learning rate (default: 1e-7)')
parser.add_argument('--margin', type=float, default=10, metavar='M',
                    help='margin for triplet loss (default: 10)')
parser.add_argument('--name', default='TripletNet', type=str,
                    help='name of experiment')
parser.add_argument('--data-path', default='triplet_data/file', type=str,
                    help='folder on s3 containing data')
parser.add_argument('--save-name', default='model-test', type=str,
                    help='the name of the saved files')
parser.add_argument('--file-totrain', default=50, type=int, metavar='N',
                    help='Number of files to train on')

args = parser.parse_args()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #resnet = models.resnet50(pretrained=True)
        resnet =  resnet50NoBN(pretrained=True)
        self.resNet = nn.Sequential(*(list(resnet.children())[:-1]))
        self.dimRed = nn.Sequential(nn.Linear(2048,1024),nn.ReLU(),nn.Linear(1024,512),nn.ReLU(),nn.Linear(512,256),
                                    nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,32))
        self.classlyr = nn.Linear(32,2)
        
    def forward(self, x):
        x = self.resNet(x)
        x = torch.squeeze(x)
        x = F.relu(self.dimRed(x))
        return self.classlyr(x)


def train(numOfFiles, numOfSamples, batchsize, model, loss_fn, optimizer, filePath):
    size = numOfFiles*numOfSamples//batchsize
    losses = []
    model.train()
    for file in range(numOfFiles):
        data = loadFile(path=filePath, num=file, s3obj=s3)
        for batch in range(numOfSamples//batchsize):
            optimizer.zero_grad()
            img, labels = getBatchData(data,batch,batchsize) 
            img, labels = img.to(device), labels.to(device)
            
            #Compute prediction error
            result = model(img)
            loss = loss_fn(result, labels) 
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    
    return np.mean(np.asarray(losses))


def test(firstFile,lastFile, numOfSamples, batchsize, model, loss_fn, filePath):
    losses = []
    model.eval()
    with torch.no_grad():
        for file in range(firstFile, lastFile):
            data = loadFile(path=filePath, num=file, s3obj=s3)
            for batch in range(numOfSamples//batchsize):
                img, labels = getBatchData(data,batch,batchsize) 
                img, labels = img.to(device), labels.to(device)

                #Compute prediction error
                results = model(img)
                loss = loss_fn(results, labels)
                losses.append(loss.item())
           
    return np.mean(np.asarray(losses))
        
            
def loadFile(path, num, s3obj):
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

def getBatchData(data,batchNum,batchsize):
    imgs1 = data[batchNum*batchsize:(batchNum+1)*batchsize,0,:,:,:]
    imgs2 = data[batchNum*batchsize:(batchNum+1)*batchsize,1,:,:,:]
    imgs3 = data[batchNum*batchsize:(batchNum+1)*batchsize,2,:,:,:]
    return imgs1, imgs2 , imgs3
        
start_time = time.perf_counter()

s3 = boto3.resource('s3',endpoint_url = 'https://s3-west.nrp-nautilus.io')
    
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


myModel = Net()
myModel.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(tnet.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

epochs = args.epochs
losses = np.empty((2,epochs))
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    losses[0,t] = train(numOfFiles=args.file_totrain, numOfSamples=1000, batchsize=args.batch_size, model=myModel, loss_fn=loss_fn, optimizer=optimizer, 
                        filePath=args.data_path)
    losses[1,t] = test(firstFile=args.file_totrain,lastFile=args.file_totrain+10, numOfSamples=1000, batchsize=args.batch_size, model=myModel,
                       loss_fn=loss_fn, filePath=args.data_path)
    print(f'Train loss: {losses[0,t]:>4f} Val loss: {losses[1,t]:>4f}')
    scheduler.step()
    
torch.save(model.state_dict(),f'model_{args.save_name}.pt')
np.save(f'loss_{args.save_name}.npy', losses, allow_pickle=True)
print('Done in {} hours'.format((time.perf_counter()-start_time)/3600))