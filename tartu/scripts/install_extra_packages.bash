#!/bin/bash

LISO="${HOME}/liso"

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate liso

pushd $LISO/config_helper
pip install --user -e .
popd

pushd $LISO/mmdetection3d
pip install --user -e .
popd

# NOTE: need gpu partition/nvcc for this
pushd $LISO/iou3d_nms
python setup.py install --user --prefix=
popd

pushd $LISO
pip install --user -e .
popd
