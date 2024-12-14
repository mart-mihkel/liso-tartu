#!/bin/bash

TARTU_RAW_ROOT="/gpfs/space/projects/ml2024"
TARTU_TARGET_DIR="/mnt/LISO_DATA_DIR/selfsupervised_OD/tartu"

conda init && . $HOME/.bashrc && conda activate liso

pushd $HOME/liso/liso/datasets/tartu
python create_tartu.py --tartu_raw_root $TARTU_RAW_ROOT --target_dir $TARTU_TARGET_DIR
