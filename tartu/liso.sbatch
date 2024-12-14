#!/bin/bash
#SBATCH -J liso-run
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

CUDA_HOME="/gpfs/space/software/cluster_software/spack/linux-centos7-x86_64/gcc-9.2.0/cuda-11.3.1-oqzddj7nezymwww6ennwec7qb6kktktw"

LISO_TARTU="${HOME}/liso/tartu"
LISO_SIF="${LISO_TARTU}/liso_tartu.sif"

module load singularity
module load cuda/11.3

singularity run --nv --bind "${CUDA_HOME},${HOME}:/mnt" $LISO_SIF $LISO_TARTU/scripts/liso.bash
