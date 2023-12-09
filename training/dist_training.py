import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import time
import os
import wandb

from model.ast_model import GWASTModel
from model.datasets import GWDataset
from config.conf import conf


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


def validation_step(ast_model, val_loader):
    val_step_losses = []
    ast_model.eval()

    with torch.no_grad():
        for data in val_loader:
            H_imgs = data[0][0].to(device)
            H_labels = data[0][1].to(device)

            with autocast():
                H_output = ast_model(H_imgs).to(device)
                H_latent = activation['ast'].to(device)

            L_imgs = data[1][0].to(device)
            L_labels = data[1][1].to(device)

            with autocast():
                L_output = ast_model(L_imgs).to(device)
                L_latent = activation['ast'].to(device)

                loss = loss_fn_1(H_output, H_labels.long()) + loss_fn_1(L_output, L_labels.long()) + 0.01 * loss_fn_2(
                    H_latent, L_latent).mean()

            val_step_losses.append(loss.item())

    return np.mean(np.asarray(val_step_losses))


def train_epoch(ast_model, train_loader, train_conf, global_step):
    ast_model.train()
    epoch_losses = []
    for data in train_loader:
        H_imgs = data[0][0].to(device)
        H_labels = data[0][1].to(device)

        if global_step <= 1000 and global_step % 50 == 0:
            warm_lr = (global_step / 1000) * lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = warm_lr
            if rank == 0:
                wandb.log({'warmup_lr': optimizer.param_groups[0]['lr']})
            print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

        with autocast():
            H_output = ast_model(H_imgs).to(device)
            H_latent = activation['ast'].to(device)

        L_imgs = data[1][0].to(device)
        L_labels = data[1][1].to(device)

        with autocast():
            L_output = ast_model(L_imgs).to(device)
            L_latent = activation['ast'].to(device)

            H_loss = loss_fn_1(H_output, H_labels.long())
            L_loss = loss_fn_1(L_output, L_labels.long())

            loss = H_loss + L_loss + train_conf['loss_fn_2_factor'] * loss_fn_2(H_latent, L_latent).mean()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_losses.append(loss.item())
        global_step += 1

    return np.mean(np.asarray(epoch_losses)), global_step


if __name__ == '__main__':
    # setup and config
    scratch_path = os.environ.get("PSCRATCH")
    data_path = os.path.join(scratch_path, f"{conf['run_config']['OGWSC_run']}_training_data_dist/")
    cfs_path = f"/global/cfs/projectdirs/{conf['project_config']['project_name']}/{conf['project_config']['logs_dir_name']}/"
    training_job_name = f"ast-training-{conf['run_config']['OGWSC_run']}-{time.strftime('%Y%m%d-%H%M%S')}"
    save_path = os.path.join(cfs_path, training_job_name)

    # distributed training setup
    device_count = torch.cuda.device_count()
    rank, n_ranks = init_workers()
    gpu = int(rank % device_count)
    device = torch.device(f'cuda:{gpu}')
    print(f"device count for rank {rank} - {device_count}")
    print(f"rank: {rank} - using GPU {gpu}")

    # create model
    ast_model = GWASTModel(**conf['model_config'])
    ast_model.to(device)
    ast_model = DDP(ast_model, device_ids=[gpu], find_unused_parameters=True)  # second param for distributed gradient sync
    ast_model.module.ast.register_forward_hook(get_activation('ast'))  # model wraped in DDP - access through .module

    # only one process writes to wandb and to save path
    if rank == 0:
        wandb.login(key=os.environ['WANDB_KEY'])
        wandb.init(
            job_type='ast_model_training',
            config=conf,
            project='gw_torch'
        )
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

    # create datasets and dataloaders
    files = [os.path.join(data_path, x) for x in os.listdir(data_path)]
    train_files = files[:int(len(files) * conf['training_config']['train_frac'])]
    val_files = files[int(len(files) * conf['training_config']['train_frac']):]

    train_ds = GWDataset(train_files)
    val_ds = GWDataset(val_files)
    train_sampler = DistributedSampler(train_ds)
    val_sampler = DistributedSampler(val_ds)

    train_loader = DataLoader(train_ds,
                              batch_size=conf['training_config']['batch_size'],
                              pin_memory=True,
                              shuffle=False,
                              sampler=train_sampler,
                              num_workers=conf['training_config']['loader_n_workers'])
    val_loader = DataLoader(val_ds,
                            batch_size=conf['training_config']['batch_size'],
                            pin_memory=True,
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=conf['training_config']['loader_n_workers'])

    print(f"train size = {len(train_loader)}")
    print(f"validation size = {len(val_loader)}")
    if rank == 0:
        wandb.log({"traing_size": len(train_loader), "validation_size": {len(val_loader)}})

    # training setup
    train_conf = conf['training_config']
    global_step = 0
    lr = train_conf['lr']
    epochs = train_conf['epochs']
    audio_trainables = [p for p in ast_model.parameters() if p.requires_grad]

    print(f'started with {lr}')
    print('Total parameter number is : {:.9f} million'.format(sum(p.numel() for p in ast_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.9f} million'.format(sum(p.numel() for p in audio_trainables) / 1e6))

    optimizer = torch.optim.Adam(audio_trainables, lr, **train_conf['optimizer_config'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **train_conf['scheduler_config'])
    scaler = GradScaler()

    loss_fn_1 = nn.CrossEntropyLoss()
    loss_fn_2 = nn.CosineSimilarity()

    train_losses = []
    val_losses = []
    best_loss = np.inf

    start_time = time.time()

    for epoch in epochs:
        epoch_loss, global_step = train_epoch(ast_model, train_loader, train_conf, global_step)
        val_loss = validation_step(ast_model, val_loader)
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        if rank == 0:
            wandb.log({"train_loss": epoch_loss, "validation_loss": val_loss})
            np.save(os.path.join(save_path, f'train_loss_ep{epoch}.npy'), np.asarray(train_losses), allow_pickle=True)
            np.save(os.path.join(save_path, f'val_loss_ep{epoch}.npy'), np.asarray(train_losses), allow_pickle=True)

            # save the best acc model
            if val_loss[-1] < best_loss:
                best_loss = val_loss[-1]
                torch.save(ast_model.state_dict(), os.path.join(save_path, "state_dict_best.pkl"))

        scheduler.step(val_losses[-1])
    dist.destroy_process_group()
    print(f'Done in {(time.time() - start_time) / 60:.2f} minutes')





