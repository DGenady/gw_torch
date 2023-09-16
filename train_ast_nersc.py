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
from random import shuffle
import os

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
    
    def __init__(self, H_files, L_files, total_length):
        self.H_files = H_files
        self.L_files = L_files
        self.H_data = []
        self.L_data = []
        self.H_labels = []
        self.L_labels = []
        self.len = total_length
        self.H_ind = 0
        self.L_ind = 0

        shuffle(self.H_files)
        shuffle(self.L_files)

    def __getitem__(self, index):
        if len(self.H_data) == 0 or len(self.L_data) == 0:
            if self.H_ind >= len(self.H_files) or self.L_ind >= len(self.L_files):
                raise StopIteration

            H_data, H_labels = self.load_data(self.H_files[self.H_ind], "H")
            L_data, L_labels = self.load_data(self.L_files[self.L_ind], "L")
            self.H_data += H_data
            self.L_data += L_data
            self.H_labels += H_labels
            self.L_labels += L_labels
        H_sample = self.H_data.pop(), self.H_labels
        L_sample = self.L_data.pop(), self.L_labels
        return H_sample, L_sample

    def __len__(self):
        return self.len
    
    def load_data(self, path, detector):
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
            assert isinstance(data, list), "pickled data is not a list"

        specs = [x['clean_spec'] for x in data]
        data = [torch.from_numpy(spec).float() for spec in specs]

        if detector == "H":
            labels = [torch.zeros_like(x) for x in data]
        elif detector == "L":
            labels = [torch.ones_like(x) for x in data]
        else:
            raise Exception(f"detector {detector} not supported")
        shuffle(data)
        return data, labels
    

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

project_name = "m4443"
scratch_path = os.environ.get("PSCRATCH")
data_path = os.path.join(scratch_path, "O3_training_data/")
cfs_path = f"/global/cfs/projectdirs/{project_name}"
training_job_name = f"ast-training-{time.strftime('%Y%m%d-%H%M%S')}"
save_path = os.path.join(cfs_path, training_job_name)
if not os.listdir(save_path):
    os.mkdir(save_path)

H_files = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.startswith("H")]
L_files = [os.path.join(data_path, x) for x in os.listdir(data_path) if x.startswith("L")]

train_frac = 0.8
specs_per_file = 5

H_train_files = H_files[:int(len(H_files)*train_frac)]
H_val_files = H_files[int(len(H_files)*train_frac):]
L_train_files = L_files[:int(len(L_files)*train_frac)]
L_val_files = L_files[int(len(L_files)*train_frac):]

train_ds_size = min(len(H_train_files)*specs_per_file, len(L_train_files)*specs_per_file)
val_ds_size = min(len(H_val_files)*specs_per_file, len(L_val_files)*specs_per_file)



global_step, epoch = 0, 0
lr = 1e-5
print(f'started with {lr}')
epochs = 40
batch_size = 120

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
    train_ds = myDataset(H_train_files, L_train_files, total_length=train_ds_size)
    val_ds = myDataset(H_val_files, L_val_files, total_length=val_ds_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    for data in train_loader:

        H_imgs = data[0][0].to(device)
        H_labels = data[0][1].to(device)

        if global_step <= 1000 and global_step % 50 == 0:
                warm_lr = (global_step / 1000) * lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

        with autocast():
            H_output = ast_mdl(H_imgs).to('cuda:0')
            H_latent = activation['ast'].to('cuda:0')

        L_imgs = data[1][0].to(device)
        L_labels = data[1][1].to(device)

        with autocast():
            L_output = ast_mdl(L_imgs).to('cuda:0')
            L_latent = activation['ast'].to('cuda:0')

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
        for data in val_loader:

            H_imgs = data[0][0].to(device)
            H_labels = data[0][1].to(device)

            with autocast():
                H_output = ast_mdl(H_imgs).to('cuda:0')
                H_latent = activation['ast'].to('cuda:0')

            L_imgs = data[1][0].to(device)
            L_labels = data[1][1].to(device)

            with autocast():
                L_output = ast_mdl(L_imgs).to('cuda:0')
                L_latent = activation['ast'].to('cuda:0')

                loss = loss_fn_1(H_output, H_labels.long()) + loss_fn_1(L_output, L_labels.long()) + 0.01*loss_fn_2(H_latent,L_latent).mean()

            losses.append(loss.item())
    
    # calculate the average loss for the validation
    val_loss.append(np.mean(np.asarray(losses)))
    
    # save the model, train loss and validation loss
    torch.save(ast_mdl.state_dict(), os.path.join(save_path, "state_dict.pkl"))
    np.save(os.path.join(save_path, f'train_loss_ep{epoch+1}.npy'), np.asarray(train_loss), allow_pickle=True)
    np.save(os.path.join(save_path, f'val_loss_ep{epoch+1}.npy'), np.asarray(val_loss), allow_pickle=True)

    
    # save best acc model
    if val_loss[-1] < best_loss:
        best_loss = val_loss[-1]
        torch.save(ast_mdl.state_dict(), os.path.join(save_path, "state_dict_best.pkl"))

    scheduler.step(val_loss[-1])
    
    # print the status
    print(f'epoch: {epoch}, train loss: {train_loss[-1]:.5f}, validation loss: {val_loss[-1]:.5f} time: {(time.time()-start_time)/60:.2f} lr: {optimizer.param_groups[0]["lr"]}')
    output_text.append(f'epoch: {epoch}, train loss: {train_loss[-1]:.5f}, validation loss: {val_loss[-1]:.5f} time: {(time.time()-start_time)/60:.2f} lr: {optimizer.param_groups[0]["lr"]}')
    
    with open(os.path.join(save_path, 'output_text.txt'),'w') as f:
        for string in output_text:
            f.write("%s\n" % string)
    
    epoch += 1

print(f'Done in {(time.time()-start_time)/60:.2f} minutes')




   
