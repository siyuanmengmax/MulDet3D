from datetime import datetime, timedelta
import os
import numpy as np
from tqdm import tqdm
from ouster.sdk import client, pcap

from ..utils.io import ensure_directory, write_bin


def main(pcap_file_path,
         json_file_path,
         start_time,
         scan_interval_seconds,
         output_dir,
         base_name):
    """
    Convert a PCAP file to binary point cloud files, with packet-loss detection.

    Args:
        pcap_file_path: Path to PCAP file.
        json_file_path: Path to sensor metadata JSON file.
        start_time: Start time used for output file naming.
        scan_interval_seconds: Expected scan interval (e.g. 0.05 for 20 Hz).
        output_dir: Output directory.
        base_name: Base name for output files.
    """
    try:
        with open(json_file_path, 'r') as f:
            metadata = client.SensorInfo(f.read())
    except Exception as e:
        raise RuntimeError(f"Failed to read metadata from {json_file_path}: {str(e)}")

    try:
        source = pcap.PcapMultiPacketReader(pcap_file_path, metadatas=[metadata])
    except Exception as e:
        raise RuntimeError(f"Failed to create PCAP source from {pcap_file_path}: {str(e)}")

    if scan_interval_seconds <= 0:
        raise ValueError("Scan interval must be positive")

    ensure_directory(output_dir)

    xyzlut = client.XYZLut(metadata)
    scans = client.ScansMulti(source).single_source(0)
    scan_interval = timedelta(seconds=scan_interval_seconds)

    total_points_original = 0
    total_points_cleaned = 0
    skipped_frames = 0
    idx = 0  # Index used for output file naming.
    processed_frames = 0
    total_detected_losses = 0

    prev_scan_timestamp = None
    packet_loss_events = []

    for scan in tqdm(scans, desc="Converting scans"):
        try:
            current_scan_timestamp_ns = scan.timestamp[0]
            current_scan_time = datetime.fromtimestamp(current_scan_timestamp_ns / 1e9)

            # Detect dropped frames by comparing the actual inter-scan interval
            # to the expected one.
            if prev_scan_timestamp is not None:
                actual_interval = (current_scan_time - prev_scan_timestamp).total_seconds()
                expected_frames = round(actual_interval / scan_interval_seconds)

                if expected_frames > 1:
                    lost_frames = expected_frames - 1
                    total_detected_losses += lost_frames
                    packet_loss_events.append({
                        'frame_idx': idx,
                        'lost_frames': lost_frames,
                        'actual_interval': actual_interval,
                        'expected_interval': scan_interval_seconds,
                        'timestamp': current_scan_time
                    })
                    idx += lost_frames  # Skip the missing frames' indices.
                    print(f"Detected {lost_frames} lost frame(s) before frame {processed_frames}. "
                          f"Actual interval: {actual_interval:.4f}s, Expected: {scan_interval_seconds:.4f}s. "
                          f"Adjusting idx from {idx - lost_frames} to {idx}")

            scan_time = start_time + idx * scan_interval

            xyz = xyzlut(scan.field(client.ChanField.RANGE)).reshape(-1, 3)
            ranges = scan.field(client.ChanField.RANGE).astype(np.float32).flatten() / 1000  # mm to m
            reflectivity = scan.field(client.ChanField.REFLECTIVITY).astype(np.float32).flatten()
            near_ir = scan.field(client.ChanField.NEAR_IR).astype(np.float32).flatten()

            # Columns: x, y, z, range, reflectivity, near_ir.
            pcd_array = np.column_stack([xyz, ranges, reflectivity, near_ir])
            total_points_original += len(pcd_array)

            non_nan_mask = ~np.isnan(xyz).any(axis=1)
            pcd_array_cleaned = pcd_array[non_nan_mask]
            unique_pcd_array = pcd_array_cleaned
            total_points_cleaned += len(unique_pcd_array)

            timestamp_str = scan_time.strftime("%Y%m%d_%H%M%S%f")
            bin_path = os.path.join(output_dir, f"{base_name}_{timestamp_str}.bin")
            write_bin(unique_pcd_array, bin_path)

            prev_scan_timestamp = current_scan_time
            idx += 1
            processed_frames += 1

        except Exception as e:
            print(f"Error processing scan {processed_frames}: {str(e)}")
            skipped_frames += 1
            continue

    print("\nPacket Loss Analysis:")
    print(f"Total processed frames: {processed_frames}")
    print(f"Total detected lost frames: {total_detected_losses}")
    print(f"Total packet loss events: {len(packet_loss_events)}")

    if packet_loss_events:
        print("\nPacket loss events details:")
        for i, event in enumerate(packet_loss_events[:10]):
            print(f"  Event {i + 1}: Lost {event['lost_frames']} frame(s) at {event['timestamp']}")
            print(f"    Actual interval: {event['actual_interval']:.4f}s vs "
                  f"Expected: {event['expected_interval']:.4f}s")
        if len(packet_loss_events) > 10:
            print(f"  ... and {len(packet_loss_events) - 10} more events")

        total_expected_frames = idx  # idx now includes all frames that should exist.
        packet_loss_rate = (total_detected_losses / total_expected_frames) * 100
        print(f"\nPacket loss rate: {packet_loss_rate:.2f}% ({total_detected_losses}/{total_expected_frames})")

    print("\nProcessing completed!")
    print("\nPoint cloud statistics:")
    print(f"Output directory: {output_dir}")
    print(f"Total original points: {total_points_original:,}")
    print(f"Total points after cleaning: {total_points_cleaned:,}")
    print(f"Removed points: {total_points_original - total_points_cleaned:,}")
    if total_points_original > 0:
        print(f"Reduction percentage: "
              f"{((total_points_original - total_points_cleaned) / total_points_original * 100):.2f}%")
    print(f"Total frames skipped due to errors: {skipped_frames}")


def validate_packet_loss_detection(pcap_file_path, json_file_path, scan_interval_seconds, max_scans=100):
    """
    Standalone diagnostic: report inter-scan interval statistics and any
    detected packet-loss events, without writing any output files.

    Args:
        pcap_file_path: Path to PCAP file.
        json_file_path: Path to sensor metadata JSON file.
        scan_interval_seconds: Expected scan interval.
        max_scans: Maximum number of scans to analyze.
    """
    try:
        with open(json_file_path, 'r') as f:
            metadata = client.SensorInfo(f.read())
    except Exception as e:
        raise RuntimeError(f"Failed to read metadata from {json_file_path}: {str(e)}")

    try:
        source = pcap.PcapMultiPacketReader(pcap_file_path, metadatas=[metadata])
    except Exception as e:
        raise RuntimeError(f"Failed to create PCAP source from {pcap_file_path}: {str(e)}")

    scans = client.ScansMulti(source).single_source(0)

    timestamps = []
    intervals = []
    prev_time = None
    scan_count = 0

    print("Validating packet loss detection logic...")
    for scan in tqdm(scans, desc="Analyzing scans"):
        if scan_count >= max_scans:
            break
        try:
            scan_timestamp_ns = scan.timestamp[0]
            scan_time = datetime.fromtimestamp(scan_timestamp_ns / 1e9)
            timestamps.append(scan_time)

            if prev_time is not None:
                interval = (scan_time - prev_time).total_seconds()
                intervals.append(interval)
                expected_frames = round(interval / scan_interval_seconds)
                if expected_frames > 1:
                    lost_frames = expected_frames - 1
                    print(f"  Frame {scan_count}: Detected {lost_frames} lost frame(s)")
                    print(f"    Interval: {interval:.4f}s, Expected frames: {expected_frames}")

            prev_time = scan_time
            scan_count += 1

        except Exception as e:
            print(f"Error analyzing scan {scan_count}: {str(e)}")
            continue

    if intervals:
        print("\nInterval statistics:")
        print(f"  Average: {np.mean(intervals):.4f} seconds")
        print(f"  Std dev: {np.std(intervals):.4f} seconds")
        print(f"  Min: {np.min(intervals):.4f} seconds")
        print(f"  Max: {np.max(intervals):.4f} seconds")
        print(f"  Expected interval: {scan_interval_seconds:.4f} seconds")


if __name__ == "__main__":
    # Example usage.
    print("=== Validating packet loss detection ===")
    validate_packet_loss_detection(
        pcap_file_path="your_file.pcap",
        json_file_path="your_metadata.json",
        scan_interval_seconds=0.05,  # 20 Hz
        max_scans=100
    )

    print("\n=== Converting with packet loss compensation ===")
    main(
        pcap_file_path="your_file.pcap",
        json_file_path="your_metadata.json",
        start_time=datetime(2024, 1, 1, 0, 0, 0),
        scan_interval_seconds=0.05,  # 20 Hz
        output_dir="output_dir",
        base_name="lidar_scan"
    )
