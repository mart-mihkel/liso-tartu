#!/usr/bin/env python3
from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path

from liso.jcp.jcp import JPCGroundRemove
from pypcd4 import PointCloud
from tqdm import tqdm

import os
import re
import numpy as np

@lru_cache(maxsize=32)
def load_tartu_pcl_image_projection_get_ground_label(pcd_file: str):
    tartu_pcl = PointCloud.from_path(pcd_file).numpy()[:, :4]

    is_ground = JPCGroundRemove(
        pcl=tartu_pcl[:, :3],
        range_img_width=2083,
        range_img_height=64,
        sensor_height=1.73,
        delta_R=1,
    )

    homog_pcl = np.copy(tartu_pcl)
    assert len(homog_pcl.shape) == 2, homog_pcl.shape
    assert homog_pcl.shape[-1] == 4, homog_pcl.shape

    homog_pcl[:, -1] = 1.0
    return tartu_pcl, homog_pcl, is_ground


def main():
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP

    argparser = ArgumentParser(description="Convert tartu pcl data to training format.")
    argparser.add_argument(
        "--target_dir",
        required=True,
        type=Path,
    )

    argparser.add_argument(
        "--tartu_raw_root",
        required=True,
        type=Path,
    )

    args = argparser.parse_args()

    target_dir = args.target_dir / "tartu_raw"
    target_dir.mkdir(parents=True, exist_ok=True)

    dates = sorted(os.listdir(args.tartu_raw_root))

    skipped_sequences = 0
    success = 0
    for date in tqdm(dates):

        try:
            stem = f"{args.tartu_raw_root}/{date}/lidar_center"
            tartu_pcd_files = [f"{stem}/{pcd}" for pcd in os.listdir(stem)]

            target_files = list(filter(lambda f: re.match(f".*{date}.*", f), os.listdir(target_dir)))
            if len(target_files) == len(tartu_pcd_files):
                skipped_sequences += 1
                continue
        except (FileNotFoundError, NotADirectoryError):
            skipped_sequences += 1
            continue

        kiss_config = KISSConfig()
        kiss_config.mapping.voxel_size = 0.01 * kiss_config.data.max_range
        odometry = KissICP(config=kiss_config)

        seq_idxs = list(range(0, len(tartu_pcd_files) - 2, 1))
        fnames = []

        for idx in tqdm(seq_idxs, leave=False):
            idx_str_t0 = Path(tartu_pcd_files[idx]).stem

            (
                pcl_t0,
                _,
                is_ground_t0,
            ) = load_tartu_pcl_image_projection_get_ground_label(
                tartu_pcd_files[idx]
            )

            # NOTE: timestamps same as above
            timestamps = np.zeros_like(pcl_t0[:, :3]).astype(np.float64)
            odometry.register_frame(
                np.copy(pcl_t0[:, :3].astype(np.float64)),
                timestamps=timestamps,
            )
            (
                pcl_t1,
                _,
                is_ground_t1,
            ) = load_tartu_pcl_image_projection_get_ground_label(
                tartu_pcd_files[idx + 1]
            )
            (
                pcl_t2,
                _,
                is_ground_t2,
            ) = load_tartu_pcl_image_projection_get_ground_label(
                tartu_pcd_files[idx + 2]
            )

            sample_name = "{0}_0_{1}".format(date, idx_str_t0)
            data_dict = {
                "pcl_t0": pcl_t0.astype(np.float32),
                "pcl_t1": pcl_t1.astype(np.float32),
                "pcl_t2": pcl_t2.astype(np.float32),
                "is_ground_t0": is_ground_t0,
                "is_ground_t1": is_ground_t1,
                "is_ground_t2": is_ground_t2,
                "name": sample_name,
            }

            target_fname = target_dir / Path(sample_name)
            fnames.append(target_fname)
            np.save(
                target_fname,
                data_dict,
            )


            if idx == seq_idxs[-1]:
                # NOTE: timestamps same as above
                timestamps = np.zeros_like(pcl_t1[:, :3]).astype(np.float64)
                odometry.register_frame(
                    np.copy(pcl_t1[:, :3].astype(np.float64)),
                    timestamps=timestamps,
                )

                # NOTE: timestamps same as above
                timestamps = np.zeros_like(pcl_t2[:, :3]).astype(np.float64)
                odometry.register_frame(
                    np.copy(pcl_t2[:, :3].astype(np.float64)),
                    timestamps=timestamps,
                )

            success += 1

        w_Ts_si = odometry.poses
        for file_idx, fname in enumerate(fnames):
            content = np.load(fname.with_suffix(".npy"), allow_pickle=True).item()
            tartu_odom_t0_t1 = (
                np.linalg.inv(w_Ts_si[file_idx]) @ w_Ts_si[file_idx + 1]
            )
            tartu_odom_t0_t2 = (
                np.linalg.inv(w_Ts_si[file_idx]) @ w_Ts_si[file_idx + 2]
            )
            tartu_odom_t1_t2 = (
                np.linalg.inv(w_Ts_si[file_idx + 1]) @ w_Ts_si[file_idx + 2]
            )
            content["kiss_odom_t0_t1"] = tartu_odom_t0_t1
            content["kiss_odom_t1_t0"] = np.linalg.inv(tartu_odom_t0_t1)
            content["kiss_odom_t0_t2"] = tartu_odom_t0_t2
            content["kiss_odom_t2_t0"] = np.linalg.inv(tartu_odom_t0_t2)
            content["kiss_odom_t1_t2"] = tartu_odom_t1_t2
            content["kiss_odom_t2_t1"] = np.linalg.inv(tartu_odom_t1_t2)

            np.save(fname, content)

    print("Skipped: {0} Success: {1}".format(skipped_sequences, success))


if __name__ == "__main__":
    main()
