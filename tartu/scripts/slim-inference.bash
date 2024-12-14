#!/bin/bash

export FORCE_CUDA="1"
export CUDA_HOME="/gpfs/space/software/cluster_software/spack/linux-centos7-x86_64/gcc-9.2.0/cuda-11.3.1-oqzddj7nezymwww6ennwec7qb6kktktw"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

conda init && . $HOME/.bashrc && conda activate liso

# pushd $HOME/liso/iou3d_nms
# python setup.py install --user --prefix=
# popd

# NOTE: change --load_checkpoint to your slim train chekpoint
pushd $HOME/liso
python3 liso/slim/cli.py --inference-only --summary-dir /mnt/LISO_DATA_DIR/flow_slim/inference_logs --load_checkpoint /mnt/LISO_DATA_DIR/flow_slim/train_logs/cb39b/20241209_195048/checkpoints/25000.pth
