"""Upload a folder of fused point clouds to Segments.ai for annotation.

Requires a Segments.ai account and API key, passed via the SEGMENTS_API_KEY
environment variable (never hardcode it in source):

    export SEGMENTS_API_KEY=your_api_key_here
    python scripts/data_preparation/upload_to_segments.py --dataset your-username/your-dataset
"""
import os
import argparse
import tempfile
from glob import glob

import numpy as np
from segments import SegmentsClient
from tqdm import tqdm


def read_bin_5_columns(file_path, num_columns=9):
    """Read a .bin file and keep only the first 5 columns (x, y, z, range, reflectivity)."""
    points = np.fromfile(file_path, dtype=np.float32)
    points = points.reshape(-1, num_columns)
    return points[:, :5]


def save_as_pcd(points, output_path):
    """Save [x, y, z, range, reflectivity] points as an ASCII .pcd file."""
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z range reflectivity
SIZE 4 4 4 4 4
TYPE F F F F F
COUNT 1 1 1 1 1
WIDTH {len(points)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(points)}
DATA ascii
"""
    with open(output_path, 'w') as f:
        f.write(header)
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]} {point[3]} {point[4]}\n")


def upload_bin_files_to_segments(bin_folder, dataset, num_columns=9):
    """Convert each .bin frame to .pcd, upload it, and register a point cloud sequence sample."""
    api_key = os.environ.get("SEGMENTS_API_KEY")
    if not api_key:
        raise RuntimeError("Set the SEGMENTS_API_KEY environment variable before running this script.")
    client = SegmentsClient(api_key)

    bin_files = sorted(glob(os.path.join(bin_folder, '*.bin')))
    print(f"Found {len(bin_files)} bin files")

    frames = []
    for i, bin_file in enumerate(tqdm(bin_files, desc="Processing bin files")):
        try:
            points = read_bin_5_columns(bin_file, num_columns=num_columns)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.pcd', delete=False) as temp_pcd:
                save_as_pcd(points, temp_pcd.name)
                filename = os.path.basename(bin_file).replace('.bin', '.pcd')
                with open(temp_pcd.name, 'rb') as f:
                    asset = client.upload_asset(f, filename=filename)
                os.unlink(temp_pcd.name)

            frame = {
                "pcd": {"url": asset.url, "type": "pcd"},
                "name": os.path.basename(bin_file).replace('.bin', ''),
                "timestamp": str(1532402927647951 + i * 100000),  # Monotonically increasing placeholder timestamp.
                "ego_pose": {
                    "position": {"x": 0, "y": 0, "z": 0},
                    "heading": {"qx": 0, "qy": 0, "qz": 0, "qw": 1},
                },
                "default_z": -1,
            }
            frames.append(frame)
            print(f"Uploaded: {filename}")

        except Exception as e:
            print(f"Error processing {bin_file}: {e}")
            continue

    if not frames:
        print("No files were successfully processed.")
        return None

    name = "lidar_point_cloud_sequence"
    attributes = {"frames": frames}
    try:
        sample = client.add_sample(dataset, name, attributes)
        print(f"\nCreated sample sequence: {name}")
        print(f"Uploaded {len(frames)} point cloud files")
        return sample
    except Exception as e:
        print(f"Error creating sample sequence: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a folder of point clouds to Segments.ai")
    parser.add_argument("--bin_folder", default="bin/merge_1000", help="Folder of fused .bin point clouds")
    parser.add_argument("--dataset", required=True, help="Segments.ai dataset identifier, e.g. 'your-username/your-dataset'")
    parser.add_argument("--num_columns", type=int, default=9, help="Number of float32 columns per point in the input .bin files")
    args = parser.parse_args()
    upload_bin_files_to_segments(args.bin_folder, args.dataset, args.num_columns)
