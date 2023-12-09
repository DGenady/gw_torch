#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J tomerh-synth-dataset-O3
#SBATCH --mail-user=hacohen0tomer@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=m4443
#SBATCH -t 01:00:00

# --- set up environment ---:
#module load python
#conda activate /global/common/software/m4443/tomerh_envs/pycbc_env

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
srun -n 20 -c 12 --cpu_bind=cores python gw_torch/make_synth_spec_dataset/main-mpi.py