# this file needs to be moved to ast/src directory to work.
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
        H_sample_clean = torch.from_numpy(sample['H']['clean_spec']).float(), torch.zeros(1)[0]  # label is 0 for H and 1 for L
        L_sample_clean = torch.from_numpy(sample['L']['clean_spec']).float(), torch.ones(1)[0]
        H_sample_signal = torch.from_numpy(sample['H']['signal_spec']).float(), torch.zeros(1)[0]  # label is 0 for H and 1 for L
        L_sample_signal = torch.from_numpy(sample['L']['signal_spec']).float(), torch.ones(1)[0]
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


ast_mdl = ast_mod(label_dim=32,
                  fstride=10, tstride=10,
                  input_fdim=256, input_tdim=256,
                  imagenet_pretrain=False,
                  audioset_pretrain=False,
                  model_size='tiny224')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
project_name = 'm4443'
GWOSC_run = 'O3'
data_path = f"/global/cfs/cdirs/{project_name}/synth_data/{GWOSC_run}_synth/"
