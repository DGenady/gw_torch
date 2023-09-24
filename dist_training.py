# this file needs to be moved to ast/src directory to work.
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from models import ASTModel
from torch.cuda.amp import autocast, GradScaler
import time
import pickle
from random import shuffle
import os


class ast_mod(nn.Module):

    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True,
                 audioset_pretrain=False, model_size='base384', verbose=True):
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

    def __init__(self, files, logfile=None):
        self.data_files = files
        self.len = len(files)
        self.logfile = logfile

    def __getitem__(self, index):
        sample_file = self.data_files[index]
        with open(sample_file, "rb") as f:
            sample = pickle.load(f)
        H_sample = torch.from_numpy(sample['H']['clean_spec']).float(), torch.zeros(1)[0]  # label is 0 for H and 1 for L
        L_sample = torch.from_numpy(sample['L']['clean_spec']).float(), torch.ones(1)[0]
        return H_sample, L_sample

    def __len__(self):
        return self.len

    def log(self, messege):
        if self.logfile is not None:
            with open(self.logfile, 'a') as f:
                f.write(messege)
        else:
            print(messege)


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook

def _get_sync_file():
    """Logic for naming sync file using slurm env variables"""
    sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH']
    os.makedirs(sync_file_dir, exist_ok=True)
    sync_file = 'file://%s/pytorch_sync.%s.%s' % (
        sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
    return sync_file

def init_workers():
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    sync_file = _get_sync_file()
    print('Setting up with sync file', sync_file)
    dist.init_process_group(backend='nccl', world_size=n_ranks, rank=rank,
                            init_method=sync_file)
    return rank, n_ranks


ast_mdl = ast_mod(label_dim=32,
                  fstride=10, tstride=10,
                  input_fdim=256, input_tdim=256,
                  imagenet_pretrain=False,
                  audioset_pretrain=False,
                  model_size='tiny224')


device_count  = torch.cuda.device_count()
rank, n_ranks = init_workers()
gpu = int(rank % device_count)
print(f"rank: {rank} - using GPU {gpu}")
device = torch.device(f'cuda:{gpu}')
ast_mdl.to(device)
ast_mdl = DDP(ast_mdl, device_ids=[gpu], find_unused_parameters=True) # second param for distributed gradient sync
ast_mdl.module.ast.register_forward_hook(get_activation('ast')) # model wraped in DDP - access through .module
loader_n_workers = 1

project_name = "m4443"
OGWSC_run = "O3"
scratch_path = os.environ.get("PSCRATCH")
data_path = os.path.join(scratch_path, f"{OGWSC_run}_training_data_dist/")
cfs_path = f"/global/cfs/projectdirs/{project_name}/ast_training_runs/"
training_job_name = f"ast-training-{OGWSC_run}-{time.strftime('%Y%m%d-%H%M%S')}"
save_path = os.path.join(cfs_path, training_job_name)

# only one process writes files
if rank == 0:
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

files = [os.path.join(data_path, x) for x in os.listdir(data_path)]
train_frac = 0.75

train_files = files[:int(len(files) * train_frac)]
val_files = files[int(len(files) * train_frac):]

print(f"train_ds size = {len(train_files)}")
print(f"val_ds size = {len(val_files)}")

global_step, epoch = 0, 0
lr = 1e-5
print(f'started with {lr}')
epochs = 40
batch_size = 1024

audio_trainables = [p for p in ast_mdl.parameters() if p.requires_grad]
print('Total parameter number is : {:.9f} million'.format(sum(p.numel() for p in ast_mdl.parameters()) / 1e6))
print('Total trainable parameter number is : {:.9f} million'.format(sum(p.numel() for p in audio_trainables) / 1e6))
trainables = audio_trainables
optimizer = torch.optim.Adam(trainables, lr, weight_decay=5e-7, betas=(0.95, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

loss_fn_1 = nn.CrossEntropyLoss()
loss_fn_2 = nn.CosineSimilarity()

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
    train_ds = myDataset(
        train_files
    )
    val_ds = myDataset(
        val_files
    )
    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              pin_memory=True,
                              shuffle=False,
                              sampler=train_sampler,
                              num_workers=loader_n_workers)
    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            pin_memory=True,
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=loader_n_workers)

    for data in train_loader:
        H_imgs = data[0][0].to(device)
        H_labels = data[0][1].to(device)

        if global_step <= 1000 and global_step % 50 == 0:
            warm_lr = (global_step / 1000) * lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = warm_lr
            print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

        with autocast():
            H_output = ast_mdl(H_imgs).to(device)
            H_latent = activation['ast'].to(device)

        L_imgs = data[1][0].to(device)
        L_labels = data[1][1].to(device)

        with autocast():
            L_output = ast_mdl(L_imgs).to(device)
            L_latent = activation['ast'].to(device)

            loss = loss_fn_1(H_output, H_labels.long()) + loss_fn_1(L_output, L_labels.long()) + 0.01 * loss_fn_2(
                H_latent, L_latent).mean()

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
                H_output = ast_mdl(H_imgs).to(device)
                H_latent = activation['ast'].to(device)

            L_imgs = data[1][0].to(device)
            L_labels = data[1][1].to(device)

            with autocast():
                L_output = ast_mdl(L_imgs).to(device)
                L_latent = activation['ast'].to(device)

                loss = loss_fn_1(H_output, H_labels.long()) + loss_fn_1(L_output, L_labels.long()) + 0.01 * loss_fn_2(
                    H_latent, L_latent).mean()

            losses.append(loss.item())

    # calculate the average loss for the validation
    val_loss.append(np.mean(np.asarray(losses)))

    # only one writer to state files
    if rank == 0:
        # save the model, train loss and validation loss
        torch.save(ast_mdl.module.state_dict(), os.path.join(save_path, "state_dict.pkl"))
        np.save(os.path.join(save_path, f'train_loss_ep{epoch + 1}.npy'), np.asarray(train_loss), allow_pickle=True)
        np.save(os.path.join(save_path, f'val_loss_ep{epoch + 1}.npy'), np.asarray(val_loss), allow_pickle=True)

        # save best acc model
        if val_loss[-1] < best_loss:
            best_loss = val_loss[-1]
            torch.save(ast_mdl.state_dict(), os.path.join(save_path, "state_dict_best.pkl"))

    scheduler.step(val_loss[-1])

    # print the status
    output_str = f'epoch: {epoch}, train loss: {train_loss[-1]:.5f}, validation loss: {val_loss[-1]:.5f} time: {(time.time() - start_time) / 60:.2f} lr: {optimizer.param_groups[0]["lr"]}'
    print(output_str)
    output_text.append(output_str)

    if rank == 0:
        with open(os.path.join(save_path, f'output_text.txt'), 'w') as f:
            for string in output_text:
                f.write("%s\n" % string)

    epoch += 1
dist.destroy_process_group()
print(f'Done in {(time.time() - start_time) / 60:.2f} minutes')
