"""File containing the logic needed to run the benchmarking of the STG-NF model.
"""

import os
import pandas as pd
import json
import random
import shutil
import numpy as np
import pathlib
import pickle
import subprocess

# Global variables
# Change these to point to the correct directories
SOURCE_POSE_DIR = './raw_data/UBnormal/poses'
SOURCE_GT_DIR = './raw_data/UBnormal/annotations/pose_level'
DST_TRAIN_POSE_DIR = './TrajREC/data/UBnormal/training/trajectories'
DST_TEST_POSE_DIR = './TrajREC/data/UBnormal/testing/trajectories'
DST_TEST_GT_DIR = './TrajREC/data/UBnormal/testing/frame_level_masks'
BENCH_MARK_DIR = './benchmark_results/TrajREC/own-Poses'

TRAINING_VIDEOS = []
for event_type in ['normal', 'abnormal']:
    with open(f'./UBnormal/scripts/{event_type}_training_video_names.txt') as f:
        TRAINING_VIDEOS.extend([line.strip() for line in f.readlines()])

TEST_VIDEOS = []
for dataset_type in ['validation', 'test']:
    for event_type in ['normal', 'abnormal']:
        with open(f'./UBnormal/scripts/{event_type}_{dataset_type}_video_names.txt') as f:
            TEST_VIDEOS.extend([line.strip() for line in f.readlines()])
CHECKPOINTS_DIR = '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/TrajREC/checkpoints'

EVAL_CMD = '''python custom_eval.py \
--trajectories '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/TrajREC/data/UBnormal/training/trajectories' \
--testdata '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/TrajREC/data/UBnormal/testing' \
--batch_size 512 \
--wandb False \
 --eval_only true \
--weights '{checkpoints_dir}/{model}' \
--video_resolution '1080x720' \
--input_length 3
'''

def remove_files(directory: str):
    """Removes all files in a directory.

    Args:
        directory (str): Target directory
    """
    shutil.rmtree(directory)

def run_command(
   command: str,
   cwd: str,
   return_output: bool = False,
   strip_response: bool = True,
) -> list:
   """Prints output of subprocess command in real-time.


   Args:
       command (string): The command to run
       return_output (bool): Return output of command as an array or not.
       strip_response (bool): Whether or not to strip whitespaces from response.


   Returns:
       list|int|None: The output of the command
   """
   output = []
   with subprocess.Popen(
       command,
       cwd=cwd,
       stdout=subprocess.PIPE,
       shell=True,
   ) as process:  # nosec
       while True:
           response = process.stdout.readline()
           if response == b"" and process.poll() is not None:
               break
           if response:
               cleaned_output = response.strip() if strip_response else response
               if return_output:
                   output.append(cleaned_output.decode("UTF-8"))
       command_response = process.poll()
   return output if return_output else command_response


def get_pose_data(pose_file: str) -> dict:
    """Fetches the pose data and returns it in a dict with the correct format

    Args:
        pose_file (str): File path to the pose data

    Returns:
        dict: Dict in the format {traj_id: [[frame, x1, y1, x2, y2, ...], ...]}
    """

    with open(pose_file) as f:
        tracking = json.load(f)
    data_dict = {}
    for traj_id, traj in tracking.items():
        csv_data = []
        for frame, data in traj.items():
            csv_record = [frame]
            key_points = data['keypoints']
            for i in range(0, len(key_points), 3):
                x = float(key_points[i])
                y = float(key_points[i+1])
                # Skip processing the confidence
                csv_record.append(x)
                csv_record.append(y)
            csv_data.append(csv_record)
        data_dict[traj_id] = csv_data
    return data_dict

def copy_gt(source_gt_dir: str, dst_gt_dir: str):
    """Copies the ground truth files to the correct directory
    and ensures that it complies with the format of the model

    Args:
        source_gt_dir (str): source of the ground truth
        dst_gt_dir (str): Target directory
    """
    remove_files(dst_gt_dir)  # Cleanup folder
    for root, dirs, files in os.walk(source_gt_dir):
        for file in files:
            if file.endswith('.npy'):
                video_name = file.split('.')[0]
                np_file = os.path.join(root,file)
                np_dest = os.path.join(dst_gt_dir, f'{video_name}_tracks.txt')
                shutil.copyfile(np_file, np_dest)


def copy_training():
    """Copies the training data to the correct directory
    and in the correct format for TrajREC to be able to use it.
    """
    remove_files(DST_TRAIN_POSE_DIR)
    for video_name in TRAINING_VIDEOS:
        pose_file = f'{SOURCE_POSE_DIR}/{video_name}_alphapose_tracked_person.json'
        pose_dict = get_pose_data(pose_file)

        for traj_id, csv_data in pose_dict.items():
            # Skip records with a short trajectory
            if len(csv_data) < 12:
                continue
            df = pd.DataFrame(csv_data)

            # Copy trajectories
            track_dir = f'{DST_TRAIN_POSE_DIR}/{video_name.replace("_", "")}_{traj_id}'
            pathlib.Path(track_dir).mkdir(parents=True, exist_ok=True)
            track_target_file = f'{track_dir}/{video_name.replace("_", "")}_{traj_id}.csv'
            # print(f'Writing trajectory to {track_target_file}')
            df.to_csv(track_target_file, index=False)


def copy_test_data(video_list):
    """Copies the given test videos to the correct directory
    and ensures it's in the correct format for TrajREC to be able to use it.
    """
    remove_files(DST_TEST_POSE_DIR)
    remove_files(DST_TEST_GT_DIR)
    for video_name in video_list:
        pose_file = f'{SOURCE_POSE_DIR}/{video_name}_alphapose_tracked_person.json'
        pose_dict = get_pose_data(pose_file)

        for traj_id, csv_data in pose_dict.items():
            # Skip records with a short trajectory
            if len(csv_data) < 12:
                continue

            df = pd.DataFrame(csv_data)

            # Copy trajectories
            track_dir = f'{DST_TEST_POSE_DIR}/{video_name.replace("_", "")}_{traj_id}'
            pathlib.Path(track_dir).mkdir(parents=True, exist_ok=True)
            track_target_file = f'{track_dir}/{video_name.replace("_", "")}_{traj_id}.csv'
            # print(f'Writing trajectory to {track_target_file}')
            df.to_csv(track_target_file, index=False)

            # Copy annotation:
            np_src_file = f'{SOURCE_GT_DIR}/{video_name}/{str(int(traj_id))}.npy'
            np_dest_dir = f'{DST_TEST_GT_DIR}/{video_name.replace("_", "")}_{traj_id}'
            pathlib.Path(np_dest_dir).mkdir(parents=True, exist_ok=True)
            np_dest_file = f'{np_dest_dir}/{video_name.replace("_", "")}_{traj_id}.npy'
            shutil.copyfile(np_src_file, np_dest_file)


def get_checkpoints(source_dir: str) -> list:
    """Get all checkpoints in the given directory

    Args:
        source_dir (str): Directory containing the checkpoints

    Raises:
        Exception: No checkpoints found

    Returns:
        list: List of the found checkpoint files
    """
    checkpoints = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.pt'):
                checkpoints.append(file)
    if not checkpoints:
        raise Exception('No checkpoints found')
    return checkpoints


def execute_benchmark(model: str, n_runs: int = 100, verbose: bool = False):
    """Executes the benchmarking of the model by running the evaluation script
    for the given amount of times.
    The results of each run is concatenated into a single CSV file.

    Args:
        model (str): Model to evaluate
        n_runs (int, optional): Number of runs to evaluate the model. Defaults to 100.
        verbose (bool, optional): Print all lines of the output of the command. Defaults to False.
    """
    pathlib.Path(BENCH_MARK_DIR).mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(n_runs):
        # Copy random subset of the test videos to the folder
        # this due to memory issues
        test_sample = random.sample(TEST_VIDEOS, 50)
        copy_test_data(test_sample)
        output = run_command(EVAL_CMD.format(model=model,
                                            checkpoints_dir=CHECKPOINTS_DIR),
                             cwd='./TrajREC',
                             return_output=True)
        for line in output:
            if verbose:
                print(line)
            if line.startswith('AVG'):
                split_line = line.split(' ')
                mse, auc = split_line[3], split_line[6]
                data_dict = {
                    'MSE': mse,
                    'AUC': auc,
                    'n_samples': test_sample,
                }

        results.append(data_dict)
    pd.DataFrame(results).to_csv(f'{BENCH_MARK_DIR}/benchmark_{model.split(".")[0]}.csv', index=False)


if __name__ == '__main__':
    #copy_training()
    checkpoints = get_checkpoints(CHECKPOINTS_DIR)
    for model in checkpoints:
        execute_benchmark(model, n_runs=100, verbose=False)
    print('Benchmarking completed')
