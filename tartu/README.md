# Liso on HPC Rocket

## Container setup 

Build the container locally, save it to a tarball and convert to singularity on hpc.

```bash
# local
docker build -t liso_tartu -f Dockerfile.tartu .
docker save liso_rocket liso_tartu.tar
scp liso_rocket.tar <username>@rocket.hpc.ut.ee:~/liso/tartu/liso_tartu.tar
```

```bash
# hpc
module load squashfs
module load singularity
singularity build liso_tartu.sif docker-archive://liso_tartu.tar
```

## Data Setup 

Note, see comment in [preprocess.sbatch](./preprocess.sbatch)

```bash
sbatch preprocess.sbatch
```

## Train SLIM, export predicted Lidar Scene Flow 

```bash
sbatch slim-train.sbatch
```

After training, to export the predicted lidar scene flow, replace `--load_checkpoint` in [slim-inference.bash](./scripts/slim-inference.bash) with your checkpoint:

```bash
sbatch slim-inference.sbatch
```

## Run LISO

Enter the directory with the exported flow .npzs `LOG_DIR/.../preds` into `liso/config/liso_config.yml` at `data.paths.tartu.slim_flow.slim_bev_120m.local=LOG_DIR/.../preds`.

```bash
sbatch liso.sbatch
```

## Tensorboard

You can copy interesting logs from LISO_DATA_DIR and open a tensorboard on them for visualizations

## Troubleshooting

On the first try, might have to run [install_extra_packages.bash](./scripts/install_extra_packages.bash) when inside the singularity container. 
This might even have to happen on a GPU partition to build `iou3d_nms`

```bash
singularity run liso_tartu.sif bash
conda init && source ~/.bashrc && conda activate liso
./scripts/install_extra_packages.bash
```
