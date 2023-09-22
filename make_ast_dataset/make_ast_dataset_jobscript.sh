#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J make_ast_dataset
#SBATCH --mail-user=hacohen0tomer@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 01:30:00
#SBATCH --account=m4443

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 32 -c 8 --cpu_bind=cores python make_ast_dataset/main-mpi.py