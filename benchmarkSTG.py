import os
import pandas as pd
import json
import random
import shutil
import numpy as np
import pathlib
import subprocess
import re
import pickle

SOURCE_VIDEO_DIR = './raw_data/UBnormal/videos'
SOURCE_POS_DIR = './raw_data/UBnormal/poses'
SOURCE_GT_DIR = './raw_data/UBnormal/annotations/frame_level'
DST_TEST_POSE_DIR = './STG-NF/data/UBnormal/pose/test'
DST_TRAIN_POSE_DIR = './STG-NF/data/UBnormal/pose/train'
DST_VIDEO_DIR = './STG-NF/data/UBnormal/videos'
DST_GT_DIR = './STG-NF/data/UBnormal/gt'
BENCH_MARK_DIR = './benchmark_results/STG-NF/own-poses'

CHECKPOINTS_DIR = '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/STG-NF/checkpoints'

TRAINING_VIDEOS = []
for event_type in ['normal', 'abnormal']:
    with open(f'./UBnormal/scripts/{event_type}_training_video_names.txt') as f:
        TRAINING_VIDEOS.extend([line.strip() for line in f.readlines()])

TEST_VIDEOS = []
for dataset_type in ['validation', 'test']:
    for event_type in ['normal', 'abnormal']:
        with open(f'./UBnormal/scripts/{event_type}_{dataset_type}_video_names.txt') as f:
            TEST_VIDEOS.extend([line.strip() for line in f.readlines()])


EVAL_CMD = '''python train_eval.py --dataset UBnormal --seg_len 16  --only_test \
--checkpoint '{checkpoints_dir}/{model}' \
--pose_path_test '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/STG-NF/data/UBnormal/pose/test' \
--pose_path_train '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/STG-NF/data/UBnormal/pose/train' \
--vid_path_train '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/raw_data/data/UBnormal/videos' \
--vid_path_test '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/raw_data/data/UBnormal/videos'
'''

def remove_files(directory: str):
    for root, dirs, files in os.walk(directory):
        for file in files:
           file_path = os.path.join(root, file)
           if os.path.isfile(file_path):
               os.remove(file_path)

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

# Copy GT:
def copy_gt(source_gt_dir: str, dst_gt_dir: str):
    # Cleanup folder:
    pathlib.Path(dst_gt_dir).mkdir(parents=True, exist_ok=True)
    remove_files(DST_GT_DIR)
    for root, dirs, files in os.walk(source_gt_dir):
        for file in files:
            if file.endswith('.npy'):
                video_name = file.split('.')[0]
                np_file = os.path.join(root,file)

                # Convert ground truth array to match the format of the STG-NF model
                gt_arr = np.load(np_file, allow_pickle=True)
                converted_array = []
                for item in gt_arr:
                    if item == 0:
                        converted_array.append(1)
                    else:
                        converted_array.append(0)
                np_dest = os.path.join(DST_GT_DIR, f'{video_name}_tracks.txt')
                np.save(np_dest, converted_array, allow_pickle=True)
                # shutil.copyfile(np_file, np_dest)
                os.rename(f"{np_dest}.npy", np_dest)   
# Copy poses
def copy_poses(videos: list, source_dir, dst_dir: str):
    pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)
    remove_files(dst_dir)
    for video_name in videos:
        file_name = f'{video_name}_alphapose_tracked_person.json'
        source_file = os.path.join(source_dir, file_name)
        dest_file = os.path.join(dst_dir,file_name)
        # print(source_file, dest_file)
        shutil.copyfile(source_file, dest_file)


def get_checkpoints(source_dir):
    checkpoints = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.tar'):
                checkpoints.append(file)
    if not checkpoints:
        raise Exception('No checkpoints found')
    return checkpoints


def execute_benchmark(model, n_runs=100):
    pathlib.Path(BENCH_MARK_DIR).mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(n_runs):
        # Copy random subset of the test videos to the folder 
        # this due to memory issues
        test_sample = random.sample(TEST_VIDEOS, 50)
        copy_poses(test_sample, SOURCE_POS_DIR, DST_TEST_POSE_DIR)
        output = run_command(EVAL_CMD.format(model=model,
                                            checkpoints_dir=CHECKPOINTS_DIR),
                             cwd='./STG-NF',
                             return_output=True)
        for line in output:
            if 'Done with' in line:
                split_line = line.split(' ')
                accuracy, samples = split_line[3], split_line[6]
                data_dict = {
                    'accuracy': accuracy,
                    'n_samples': samples,
                    'scenes_used': test_sample
                }
                results.append(data_dict)
                break
        else:
            for line in output:
                print(line)
            raise Exception('Model failed to execute correctly')
    pd.DataFrame(results).to_csv(f'{BENCH_MARK_DIR}/benchmark_{model.split(".")[0]}.csv', index=False)


if __name__ == '__main__':
    # copy_gt(SOURCE_GT_DIR, DST_GT_DIR)
    # copy_training()
    checkpoints = get_checkpoints(CHECKPOINTS_DIR)
    checkpoints = ['ShanghaiTech_85_9.tar']
    for model in checkpoints:
        execute_benchmark(model, n_runs=100)
    print('Benchmarking completed')