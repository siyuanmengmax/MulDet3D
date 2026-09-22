"""Render a PCAP capture's range/reflectivity/near-IR channels to PNG frames and MP4 videos.

Useful for quickly inspecting a raw recording before running the full
.bin conversion pipeline.
"""
import os
import sys

import numpy as np
import cv2
from ouster.sdk import client
from ouster.sdk.client import ChanField
from ouster.sdk import open_source
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.io import ensure_directory


def refine_image(raw_image, enhance=True, denoise=True, low_percent=1, high_percent=99, brightness=1.7, kernel_size=3):
    """
    Process the image by enhancing contrast and applying noise removal.
    Args:
        raw_image: Input image to be processed
        enhance: Whether to apply contrast enhancement
        denoise: Whether to apply noise removal
        low_percent: Lower percentile for contrast stretching
        high_percent: Upper percentile for contrast stretching
        brightness: Brightness multiplier
        kernel_size: Size of median blur kernel for noise removal
    Returns:
        processed image
    """
    if enhance:
        non_zero_mask = raw_image > 0
        if np.any(non_zero_mask):
            valid_pixels = raw_image[non_zero_mask]
            min_val, max_val = np.percentile(valid_pixels, [low_percent, high_percent])
            if max_val > min_val:
                normalized = np.zeros_like(raw_image)
                normalized[non_zero_mask] = 255 * (valid_pixels - min_val) / (max_val - min_val)
                raw_image = np.clip(normalized * brightness, 0, 255)
    if denoise:
        raw_image = cv2.medianBlur(raw_image.astype(np.uint8), kernel_size)
    return raw_image


def setup_video_writers(field_names, output_dir, fps=10, first_image=None):
    """
    Set up video writers for each channel.

    Args:
        field_names: List of field names
        output_dir: Base output directory
        fps: Frames per second for the video
        first_image: Sample image to determine dimensions

    Returns:
        Dictionary of video writers
    """
    video_writers = {}
    if first_image is not None:
        height, width = first_image.shape[:2]
    else:
        height, width = 128, 1024
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for field_name in field_names:
        video_path = os.path.join(output_dir, f"{field_name}.mp4")
        video_writers[field_name] = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=True)
    return video_writers


def main(pcap_path, metadata_path, output_dir, fields=[ChanField.RANGE, ChanField.REFLECTIVITY, ChanField.NEAR_IR],
         field_names=["range", "reflectivity", "near_ir"], save_images=True, save_videos=True, fps=10):
    """
    Process a pcap file and save per-channel images and videos.
    Args:
        pcap_path: Path to the pcap file
        metadata_path: Path to the metadata json file
        output_dir: Base output directory
        fields: List of LiDAR channel fields to process
        field_names: List of field names corresponding to fields
        save_images: Whether to save individual frame images
        save_videos: Whether to save videos
        fps: Frames per second for the videos
    """
    ensure_directory(output_dir)
    field_folders = {}
    if save_images:
        for field_name in field_names:
            field_folder = os.path.join(output_dir, field_name)
            ensure_directory(field_folder)
            field_folders[field_name] = field_folder

    scan_source = open_source(pcap_path)
    with open(metadata_path, "rb") as metadata_file:
        metadata = client.SensorInfo(metadata_file.read())

    video_writers = None
    try:
        frame_count = 0
        for scan in tqdm(scan_source, desc="Processing frames"):
            processed_images = {}
            for field, field_name in zip(fields, field_names):
                data = scan.field(field).astype(np.float32)
                image = client.destagger(metadata, data)
                refined_image = refine_image(image)
                bgr_image = cv2.cvtColor(refined_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                processed_images[field_name] = bgr_image
                if save_images:
                    frame_filename = os.path.join(field_folders[field_name], f"frame_{frame_count:05d}.png")
                    cv2.imwrite(frame_filename, refined_image)

            if save_videos and video_writers is None and processed_images:
                first_image = next(iter(processed_images.values()))
                video_writers = setup_video_writers(field_names, output_dir, fps, first_image)
            if save_videos and video_writers:
                for field_name, image in processed_images.items():
                    video_writers[field_name].write(image)

            frame_count += 1
        print(f'Done: generated {frame_count} frames')
    except Exception as e:
        print(f"Error processing data: {str(e)}")
    finally:
        if video_writers:
            for writer in video_writers.values():
                writer.release()


if __name__ == "__main__":
    output_dir = "bin/preview_images"
    pcap_file_path = "pcap/one/sensor_one.pcap"
    json_file_path = "pcap/one/sensor_one.json"
    main(pcap_file_path, json_file_path, output_dir, save_images=True, save_videos=True, fps=20)
