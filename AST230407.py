import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import timm
from models import ASTModel
from torch.cuda.amp import autocast,GradScaler
import time
import boto3
import pickle

class GWDATA:
    
    def __init__(self, H=None, L=None, times=None):
        self.H = H
        self.L = L
        self.times = times
     
        
    def save(self,path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)



class ast_mod(nn.Module):
    
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=True):
        
        def get_sinusoid_encoding(n_position, d_hid):
            ''' Sinusoid position encoding table '''

            def get_position_angle_vec(position):
                return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

            sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
            sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
            sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

            return torch.FloatTensor(sinusoid_table).unsqueeze(0)
        
        super(ast_mod, self).__init__()
        self.ast = ASTModel(label_dim=label_dim, fstride=fstride, tstride=tstride,
                            input_fdim=input_fdim, input_tdim=input_tdim, imagenet_pretrain=imagenet_pretrain,
                            audioset_pretrain=audioset_pretrain, model_size=model_size, verbose=verbose)
        self.last_layer = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 2))
        self.ast.pos_embed = nn.Parameter(get_sinusoid_encoding(self.ast.v.patch_embed.num_patches + 2,
                                                   self.ast.original_embedding_dim), requires_grad=False)
            
    def forward(self, x):
        x = self.ast(x) 
        x = self.last_layer(x)
        return x
    
    
class myDataset(Dataset):
    
    def __init__(self, file,path):
        self.len = None
        self.file = file
        self.data = self.load_data(path)
            
    def __getitem__(self, index):
        return ((self.data[0][0][index],self.data[0][1][index]),(self.data[1][0][index],self.data[1][1][index]))

    def __len__(self):
        return self.len
    
    def load_data(self,path):
        
        with open(path+self.file,'rb') as f:
            data = pickle.load(f)
            
        H = torch.from_numpy(data.H['spectrograms']).float()
        L = torch.from_numpy(data.L['spectrograms']).float()
        
        H_labels = torch.zeros(H.size(0))
        L_labels = torch.zeros(L.size(0)) + 1
        
        inds = np.arange(L.size(0))
        np.random.shuffle(inds)
        
        L = L[inds]
        
        self.len = H.size(0)
    
        return (H,H_labels), (L,L_labels)
    

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

ast_mdl = ast_mod(label_dim=32, 
                  fstride=10, tstride=10, 
                  input_fdim=256, input_tdim=256, 
                  imagenet_pretrain=True, 
                  audioset_pretrain=False, 
                  model_size='tiny224')
                  
                  
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    ast_mdl  = nn.DataParallel(ast_mdl)

ast_mdl.to(device)
ast_mdl.module.ast.register_forward_hook(get_activation('ast'))

s3 = boto3.resource('s3', endpoint_url = 'https://s3-west.nrp-nautilus.io')   
my_bucket = s3.Bucket('tau-astro')
files_to_down = []

for object_summary in my_bucket.objects.filter(Prefix="gdevit/gw_data/O1/Both"):
    files_to_down.append(object_summary.key)
    
files = []

files_to_down.remove('gdevit/gw_data/O1/Both/saved_segments.txt')

s3 = boto3.client('s3', endpoint_url = 'https://s3-west.nrp-nautilus.io')   

for file in files_to_down[:23]:
    save_name = file.split('/')[-1]
    s3.download_file('tau-astro', file, 'data/'+save_name)
    files.append(save_name)
    
train_files = files[:23]
val_files = files[20:23]


save_name = '230407'
path = 'data/'

global_step, epoch = 0, 0
lr = 1e-5
print(f'started on 7/4/23 with {lr}')
epochs = 40
batch_size = 60

audio_trainables = [p for p in ast_mdl.parameters() if p.requires_grad]
print('Total parameter number is : {:.9f} million'.format(sum(p.numel() for p in ast_mdl.parameters()) / 1e6))
print('Total trainable parameter number is : {:.9f} million'.format(sum(p.numel() for p in audio_trainables) / 1e6))
trainables = audio_trainables
optimizer = torch.optim.Adam(trainables, lr, weight_decay=5e-7, betas=(0.95, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

loss_fn_1 = nn.CrossEntropyLoss()
loss_fn_2 = nn.CosineSimilarity()

epoch += 1

train_loss = []
val_loss = []
best_loss = np.inf

scaler = GradScaler()

start_time = time.time()

output_text = []

while epoch < epochs + 1:
    # Training step
    losses = []
    ast_mdl.train()
    
    np.random.shuffle(train_files)
    
    for file in train_files:

        train_loader = DataLoader(dataset=myDataset(file=file,path=path), batch_size=batch_size, shuffle=True)

        for data in train_loader:
            
            H_imgs = data[0][0].to(device)
            H_labels = data[0][1].to(device)
            
            if global_step <= 1000 and global_step % 50 == 0:
                    warm_lr = (global_step / 1000) * lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = warm_lr
                    print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            with autocast():
                H_output = ast_mdl(H_imgs)
                H_latent = activation['ast']
             
            L_imgs = data[1][0].to(device)
            L_labels = data[1][1].to(device)
            
            with autocast():
                L_output = ast_mdl(L_imgs)
                L_latent = activation['ast']
                
                loss = loss_fn_1(H_output, H_labels.long()) + loss_fn_1(L_output, L_labels.long()) + 0.01*loss_fn_2(H_latent,L_latent).mean()
          
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            losses.append(loss.item())
            global_step += 1

    train_loss.append(np.mean(np.asarray(losses)))
    # Validation step
    losses = []
    ast_mdl.eval()
    
    with torch.no_grad():
        for file in val_files:

            val_loader = DataLoader(dataset=myDataset(file=file,path=path), batch_size=batch_size, shuffle=True)   

            for data in val_loader:
                
                H_imgs = data[0][0].to(device)
                H_labels = data[0][1].to(device)

                with autocast():
                    H_output = ast_mdl(H_imgs)
                    H_latent = activation['ast']
             
                L_imgs = data[1][0].to(device)
                L_labels = data[1][1].to(device)
            
                with autocast():
                    L_output = ast_mdl(L_imgs)
                    L_latent = activation['ast']
                
                    loss = loss_fn_1(H_output, H_labels.long()) + loss_fn_1(L_output, L_labels.long()) + 0.01*loss_fn_2(H_latent,L_latent).mean()
                
                losses.append(loss.item())
    
    # calculate the average loss for the validation
    val_loss.append(np.mean(np.asarray(losses)))
    
    # save the model, train loss and validation loss
    torch.save(ast_mdl.state_dict(), f'{save_name}.pt')
    np.save(f'{save_name}_train_loss.npy', np.asarray(train_loss), allow_pickle=True)
    np.save(f'{save_name}_val_loss.npy', np.asarray(val_loss), allow_pickle=True)
    
    
    # save best acc model
    if val_loss[-1] < best_loss:
        best_loss = val_loss[-1]
        torch.save(ast_mdl.state_dict(), f'{save_name}_best.pt')
    
    scheduler.step(val_loss[-1])
    
    # print the status
    print(f'epoch: {epoch}, train loss: {train_loss[-1]:.5f}, validation loss: {val_loss[-1]:.5f} time: {(time.time()-start_time)/60:.2f} lr: {optimizer.param_groups[0]["lr"]}')
    output_text.append(f'epoch: {epoch}, train loss: {train_loss[-1]:.5f}, validation loss: {val_loss[-1]:.5f} time: {(time.time()-start_time)/60:.2f} lr: {optimizer.param_groups[0]["lr"]}')
    
    with open('output_text.txt','w') as f:
        for string in output_text:
            f.write("%s\n" % string)
    
    epoch += 1
    trys = 0
    try:
        s3.upload_file(f'{save_name}.pt', 'tau-astro', f'gdevit/model/{save_name}/{save_name}.pt' )
        s3.upload_file(f'{save_name}_best.pt', 'tau-astro', f'gdevit/model/{save_name}/{save_name}_best.pt' )
        s3.upload_file(f'{save_name}_train_loss.npy', 'tau-astro', f'gdevit/model/{save_name}/{save_name}_train_loss.npy' )
        s3.upload_file(f'{save_name}_val_loss.npy', 'tau-astro', f'gdevit/model/{save_name}/{save_name}_val_loss.npy' )
        s3.upload_file('output_text.txt', 'tau-astro', f'gdevit/model/output_text.txt' )
    except: 
        trys += 1
        print('did not upload')
        
if trys == epochs + 1:
    print('didnt upload at all')
    
s3.upload_file(f'{save_name}.pt', 'tau-astro', f'gdevit/model/{save_name}/{save_name}.pt' )
s3.upload_file(f'{save_name}_best.pt', 'tau-astro', f'gdevit/model/{save_name}/{save_name}_best.pt' )
s3.upload_file(f'{save_name}_train_loss.npy', 'tau-astro', f'gdevit/model/{save_name}/{save_name}_train_loss.npy' )
s3.upload_file(f'{save_name}_val_loss.npy', 'tau-astro', f'gdevit/model/{save_name}/{save_name}_val_loss.npy' )


print(f'Done in {(time.time()-start_time)/60:.2f} minutes')




   
