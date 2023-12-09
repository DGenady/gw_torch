#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J train-O3-ast-dist
#SBATCH --mail-user=hacohen0tomer@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 03:30:00
#SBATCH -A m4443

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WANDB_KEY=''

# before running ther script, activate conda env:
# conda activate pytorch_env (clone of module pytorch, pytorch downgraded to 1.13, with timm==0.4.5 and wget installed)
# for example:
# conda activate /global/homes/t/tomerh/envs/pytorch_pretrained_env


#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1
srun -n 4 -c 32 --cpu_bind=cores -G 4 --gpu-bind=single:1  python /global/homes/t/tomerh/gw_torch/training/dist_training.py