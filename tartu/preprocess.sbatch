#!/bin/bash
#SBATCH -J liso-preprocess
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --partition=amd

# NOTE: if the job runs out of time while processing a date
# we will end up with some saved files that don't have kiss_icp
# odometry estimations, these should be deleted
#
# cd ~/LISO_DATA_DIR/selfsupervised_OD/tartu/tartu_raw
# # list number of proceced files by date prefix
# ls | awk -F'_' '{print $1}' | sort | uniq -c
# rm [date-prefix-where-proccessing-was-cut-off]*
#
# running the job again should skip dates that have already
# been fully processed

TARTU_RAW_ROOT="/gpfs/space/projects/ml2024"

LISO_TARTU="${HOME}/liso/tartu"
LISO_SIF="${LISO_TARTU}/liso_tartu.sif"

module load singularity

singularity run --bind "${TARTU_RAW_ROOT},${HOME}:/mnt" $LISO_SIF $LISO_TARTU/scripts/preprocess.bash
