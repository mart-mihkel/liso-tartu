#!/bin/bash

export FORCE_CUDA="1"
export CUDA_HOME="/gpfs/space/software/cluster_software/spack/linux-centos7-x86_64/gcc-9.2.0/cuda-11.3.1-oqzddj7nezymwww6ennwec7qb6kktktw"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

conda init && . $HOME/.bashrc && conda activate liso

pip install --user -e moviepy

pushd $HOME/liso
python3 liso/kabsch/liso_cli.py --summary-dir /mnt/LISO_DATA_DIR/liso/train_logs -c tartu bev_100m_512 centerpoint batch_size_four liso
