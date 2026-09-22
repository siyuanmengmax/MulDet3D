"""Strip fused point clouds down to [x, y, z, reflectivity] for annotation upload.

Most labeling tools (e.g. Segments.ai) only need position and reflectivity;
this reformats a folder of fused .bin frames accordingly before upload.
"""
import os
import sys
import argparse
from glob import glob

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.io import ensure_directory, read_bin, write_bin


def main(args):
    print("Processing point cloud files for annotation...")
    all_bin_files = sorted(glob(os.path.join(args.input_folder, '*.bin')))
    ensure_directory(args.output_folder)
    print(f"Found {len(all_bin_files)} bin files in input folder")

    selected_files = all_bin_files[args.start_frame:args.end_frame:args.frame_step]
    for bin_file in tqdm(selected_files):
        pcd_array = read_bin(bin_file, len(args.columns))
        x = pcd_array[:, args.columns.index('x')]
        y = pcd_array[:, args.columns.index('y')]
        z = pcd_array[:, args.columns.index('z')]
        reflectivity = pcd_array[:, args.columns.index('reflectivity')]
        new_pcd_array = np.column_stack([x, y, z, reflectivity])
        filename = os.path.basename(bin_file)
        output_path = os.path.join(args.output_folder, filename)
        write_bin(new_pcd_array, output_path)
    print(f"Wrote {len(selected_files)} bin files to {args.output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reformat point clouds to [x, y, z, reflectivity] for annotation upload")
    parser.add_argument("--input_folder", default="bin/merge_1000", help="Folder containing input point cloud files")
    parser.add_argument("--output_folder", default="bin/merge_1000_for_upload", help="Folder for reformatted files")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame index")
    parser.add_argument("--frame_step", type=int, default=1, help="Process every nth frame")
    columns = ['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'intensity', 'flags', 'dynamic_mask', 'reliability',
               'lidar_source']
    parser.add_argument("--columns", default=columns, help="Columns in the input bin files")
    args = parser.parse_args()
    main(args)
