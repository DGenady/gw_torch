#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 2
#SBATCH -q regular
#SBATCH -J train-GW-ast-O3
#SBATCH --mail-user=hacohe0tomer@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 04:30:00
#SBATCH -A m4443

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# before running ther script, activate conda env:
# conda activate pytorch_env (clone of module pytorch with timm==0.4.5 and wget installed)
#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1
srun -n 2 -c 64 --cpu_bind=cores -G 2 --gpu-bind=none  python /global/homes/t/tomerh/ast/src/train_ast_nersc.py