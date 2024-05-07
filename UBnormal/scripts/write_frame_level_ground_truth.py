"""This script creates the ground truth on frame level for UBnormal
"""

import pdb

import cv2
import numpy as np
import os


def get_number_of_frames(video_path):
    video = cv2.VideoCapture(video_path)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    video.release()
    return total


def write_frame_level_ground_truth(output_folder_base, video_dir, filenames, is_abnormal=True):
    video_names = np.loadtxt(filenames, dtype=str)
    video_names.sort()

    for video_idx, video_name in enumerate(video_names):
        os.makedirs(f"{output_folder_base}/{video_name}", exist_ok=True)
        print(video_idx, video_name)
        scene_name = video_name.split("_")[2]

        video_path = os.path.join(video_dir, f'Scene{scene_name}', video_name + ".mp4")
        num_frames = get_number_of_frames(video_path)
        ground_truth = np.zeros(num_frames)

        gt_path = os.path.join(output_folder_base, video_name, "ground_truth_frame_level.txt")
        if is_abnormal:
            tracks_path = os.path.join(video_dir, f'Scene{scene_name}', video_name + "_annotations", video_name + "_tracks.txt")
            tracks_video = np.loadtxt(tracks_path, delimiter=",")
            if tracks_video.ndim == 1:
                tracks_video = [tracks_video]

            for track in tracks_video:
                ground_truth[int(track[1]): int(track[2] + 1)] = 1

            np.savetxt(gt_path, ground_truth)

        else:
            np.savetxt(gt_path, ground_truth)

for data_type in ["training", "validation", "test"]:
    # path to the output folder
    output_folder_base = f"/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/UBnormal/data/frame_gt/{data_type}"
    # path to the data set
    video_dir = "/mnt/c/Users/ruben/Downloads/UBnormal_data"
    # path to the txt file with the video names
    for type in ["normal", "abnormal"]:
        filenames = f"./{type}_{data_type}_video_names.txt"
        # if the list with the video names are for the normal videos set is_abnormal to False, otherwise set it to True.
        is_abnormal = type=="abnormal"
        write_frame_level_ground_truth(output_folder_base, video_dir, filenames, is_abnormal=is_abnormal)
