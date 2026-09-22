import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing import time_alignment


def main(args):
    time_alignment.main(
        args.input_folder1_path,
        args.input_folder2_path,
        args.output_log1_path,
        args.output_log2_path,
        args.start_time,
        args.end_time,
        args.time_offset_us,
        args.max_time_diff_us
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter and time-align PCD files from two sensors")
    parser.add_argument("--input_folder1_path", default="bin/one", help="Input folder for source 1")
    parser.add_argument("--input_folder2_path", default="bin/two", help="Input folder for source 2")
    parser.add_argument("--output_log1_path", default="bin/one.txt", help="Output log file for source 1")
    parser.add_argument("--output_log2_path", default="bin/two.txt", help="Output log file for source 2")
    parser.add_argument("--start_time", default="00:00:00", help="Start time in HH:MM:SS format")
    parser.add_argument("--end_time", default="23:59:59", help="End time in HH:MM:SS format")
    parser.add_argument("--time_offset_us", type=int, default=0, help="Expected time offset in microseconds")
    parser.add_argument("--max_time_diff_us", type=int, default=50000, help="Maximum allowed time difference in microseconds")
    args = parser.parse_args()
    main(args)
