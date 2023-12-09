import numpy as np
import torch
import torch.nn as nn

import sys

sys.path.append(r"/global/homes/t/tomerh/ast/src/")  # path of ast repo
from models import ASTModel


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class GWASTModel(nn.Module):
    def __init__(
            self,
            label_dim=527,
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=1024,
            imagenet_pretrain=True,
            audioset_pretrain=False,
            model_size='base384',
            verbose=True
    ):
        super(GWASTModel, self).__init__()
        self.ast = ASTModel(label_dim=label_dim, fstride=fstride, tstride=tstride,
                            input_fdim=input_fdim, input_tdim=input_tdim, imagenet_pretrain=imagenet_pretrain,
                            audioset_pretrain=audioset_pretrain, model_size=model_size, verbose=verbose)
        self.last_layer = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 2))
        self.ast.pos_embed = nn.Parameter(
            get_sinusoid_encoding(self.ast.v.patch_embed.num_patches + 2, self.ast.original_embedding_dim),
            requires_grad=False
        )

    def forward(self, x):
        x = self.ast(x)
        x = self.last_layer(x)
        return x
