import os
from datetime import datetime, timedelta
from tqdm import tqdm


def parse_time_from_filename(filename):
    """Parse a timestamp from a `..._YYYYMMDD_HHMMSSffffff.bin` style filename."""
    try:
        parts = filename.split('_')
        date_str = parts[-2]   # e.g. 20240807
        time_str = parts[-1].split('.')[0]  # e.g. 110527000000
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        microsecond = int(time_str[6:])
        return datetime(year, month, day, hour, minute, second, microsecond)
    except (IndexError, ValueError) as e:
        print(f"Error parsing time from filename {filename}: {e}")
        return None


def main(input_log1, input_log2, output_log1, output_log2,
         start_time="17:00:00", end_time="18:00:00",
         time_offset_us=150000, max_time_diff_us=1000):
    """
    Time-align two sensors' frame sequences by filename timestamp.

    Reads all .bin files in `input_log1`/`input_log2`, restricts to a time
    window [start_time, end_time), corrects for a known fixed clock offset
    between the two sensors (`time_offset_us`), and greedily matches each
    sensor-1 frame to its nearest sensor-2 frame within `max_time_diff_us`.
    Writes the matched filename pairs to `output_log1`/`output_log2`.
    """
    print("\nChecking input files...")
    if not os.path.exists(input_log1) or not os.path.exists(input_log2):
        print("Error: Input files not found!")
        return

    print("\nReading input files...")
    files1 = sorted([file for file in os.listdir(input_log1) if file.endswith('.bin')])
    files2 = sorted([file for file in os.listdir(input_log2) if file.endswith('.bin')])
    print(f"Read {len(files1)} files from {input_log1}")
    print(f"Read {len(files2)} files from {input_log2}")

    try:
        start_hour, start_minute, start_second = map(int, start_time.split(':'))
        end_hour, end_minute, end_second = map(int, end_time.split(':'))
        start_dt = parse_time_from_filename(files1[0]).replace(
            hour=start_hour, minute=start_minute, second=start_second, microsecond=0)
        end_dt = parse_time_from_filename(files1[0]).replace(
            hour=end_hour, minute=end_minute, second=end_second, microsecond=0)
        print(f"\nTime range: {start_time} - {end_time}")
    except ValueError as e:
        print(f"Error parsing time range: {e}")
        return

    print("\nCreating time indices...")
    time_dict1 = {}  # Timestamp index for source 1.
    time_dict2 = {}  # Timestamp index for source 2.
    for file1 in tqdm(files1, desc="Processing source 1"):
        time1 = parse_time_from_filename(file1)
        if time1 and start_dt <= time1 < end_dt:
            time_dict1[time1] = file1
    for file2 in tqdm(files2, desc="Processing source 2"):
        time2 = parse_time_from_filename(file2)
        if time2:
            # Source 2's clock lags by time_offset_us; correct before comparing.
            adjusted_time = time2 - timedelta(microseconds=time_offset_us)
            if start_dt <= adjusted_time < end_dt:
                time_dict2[adjusted_time] = file2
    print("\nValid timestamps found:")
    print(f"Source 1: {len(time_dict1)}")
    print(f"Source 2: {len(time_dict2)}")

    valid_pairs = []
    for time1, file1 in tqdm(time_dict1.items(), desc="Finding matches"):
        target_time = time1
        closest_time = None
        min_diff = float('inf')
        for time2 in time_dict2.keys():
            diff = abs((time2 - target_time).total_seconds() * 1_000_000)
            if diff < min_diff and diff <= max_time_diff_us:
                min_diff = diff
                closest_time = time2
        if closest_time is not None:
            valid_pairs.append((file1, time_dict2[closest_time], min_diff))

    if not valid_pairs:
        print("\nWarning: No valid pairs found!")
        return

    valid_pairs.sort(key=lambda x: x[2])
    print(f"\nWriting {len(valid_pairs)} pairs to output files...")
    with open(output_log1, 'w') as f1, open(output_log2, 'w') as f2:
        for file1, file2, _ in valid_pairs:
            f1.write(f"{file1}\n")
            f2.write(f"{file2}\n")

    print("\nExample matches (first 5 pairs):")
    for file1, file2, diff in valid_pairs[:5]:
        time1 = parse_time_from_filename(file1)
        time2 = parse_time_from_filename(file2)
        print(f"\nActual time difference: {diff:.2f}us")
        print(f"  SRC1: {time1.strftime('%H:%M:%S.%f')} - {os.path.basename(file1)}")
        print(f"  SRC2: {time2.strftime('%H:%M:%S.%f')} - {os.path.basename(file2)}")

    diffs = [diff for _, _, diff in valid_pairs]
    print("\nTime difference statistics (microseconds):")
    print(f"  Minimum: {min(diffs):.2f}")
    print(f"  Maximum: {max(diffs):.2f}")
    print(f"  Average: {sum(diffs) / len(diffs):.2f}")
    print("\nResults written to:")
    print(f"  - {output_log1}")
    print(f"  - {output_log2}")
    print(f"Total pairs written: {len(valid_pairs)}")
