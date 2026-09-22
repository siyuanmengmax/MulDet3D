"""Convert sensor 2's raw PCAP capture into per-frame .bin point clouds."""
import os
import sys
from datetime import datetime
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing import pcap_to_bin
from src.utils.io import ensure_directory


def main(args):
    ensure_directory(args.output_folder_path)
    pcap_to_bin.main(
        pcap_file_path=args.pcap_file_path,
        json_file_path=args.json_file_path,
        output_dir=args.output_folder_path,
        start_time=args.start_time,
        scan_interval_seconds=args.scan_interval_seconds,
        base_name=args.base_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert sensor 2's PCAP capture to .bin point clouds")
    parser.add_argument('--pcap_file_path', default='pcap/two/sensor_two.pcap', help='Path to the PCAP file')
    parser.add_argument('--json_file_path', default='pcap/two/sensor_two.json', help='Path to the sensor metadata JSON file')
    parser.add_argument('--output_folder_path', default='bin/two', help='Output directory for .bin files')
    parser.add_argument('--start_time', default=datetime(2024, 1, 1, 0, 0, 0), help='Start time used for output file naming')
    parser.add_argument('--scan_interval_seconds', type=float, default=1 / 20, help='Scan interval in seconds (e.g. 1/20 for 20 Hz)')
    parser.add_argument('--base_name', default='two', help='Base name for output files')
    args = parser.parse_args()
    main(args)
