#!/bin/bash
#SBATCH -N 3
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J make_ast_dataset_O1
#SBATCH --mail-user=gdevit18@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 04:30:00
#SBATCH --account=m4443

# --- set up environment ---:
#module load python
#conda activate /global/common/software/m4443/tomerh_envs/gwpy_env

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 384 -c 2 --cpu_bind=cores python /global/homes/g/gdevit/gw_torch/make_spec_dataset/main-mpi.py
