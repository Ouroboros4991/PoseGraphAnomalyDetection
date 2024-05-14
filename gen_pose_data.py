# This script is based on the following script: https://github.com/orhir/STG-NF/blob/main/gen_data.py
import argparse
import json
import os
import shutil
import logging
from posixpath import basename


def convert_data_format(data, split='None'):
    if split == 'testing':
        num_digits = 3
    elif split == 'training':
        num_digits = 4
    elif split == 'None':
        num_digits = 4

    data_new = dict()
    for item in data:
        frame_idx_str = item['image_id'][:-4]  # '0.jpg' -> '0'
        frame_idx_str = frame_idx_str.zfill(num_digits)  # '0' -> '000'
        person_idx_str = str(item['idx'])
        keypoints = item['keypoints']
        scores = item['score']
        if not person_idx_str in data_new:
            data_new[person_idx_str] = {frame_idx_str: {'keypoints': keypoints, 'scores': scores}}
        else:
            data_new[person_idx_str][frame_idx_str] = {'keypoints': keypoints, 'scores': scores}

    return data_new


def read_convert_write(in_full_fname, out_full_fname):
    # Read results file
    with open(in_full_fname, 'r') as f:
        data = json.load(f)

    # Convert reults file format
    data_new = convert_data_format(data)

    # 3. Write
    save = True  # False
    if save:
        with open(out_full_fname, 'w') as f:
            json.dump(data_new, f)


def create_command(alphapose_dir, video_filename, out_dir, is_video=False):
    command_args = {'cfg': os.path.join(alphapose_dir, 'configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml'),
                    'checkpoint': os.path.join(alphapose_dir, 'pretrained_models/fast_421_res152_256x192.pth'),
                    'outdir': out_dir}

    command = f"python {os.path.join(alphapose_dir, 'scripts/demo_inference.py')}"
    # command = "python scripts/demo_inference.py"

    # loop over command line argument and add to command
    for key, val in command_args.items():
        command += ' --' + key + ' ' + val
    if is_video:
        command += ' --video ' + video_filename
    else:
        command += ' --indir ' + video_filename
    command += ' --sp'  # Torch Re-ID Track
    command += ' --pose_track'  # Torch Re-ID Track
    command += ' --detector yolox-x'  # Simple MOT Track
    # print(command)
    return command


def main():
    # parse command line
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True, help='Target dataset such as UBI_FIGHTS or UBnormal. This dataset should be located in the data folder')
    args = ap.parse_args()
    alphapose_dir = os.path.join(os.getcwd(), 'AlphaPose')
    video_dir = os.path.join(os.getcwd(), f'raw_data/{args.dataset}/videos')
    output_dir = os.path.join(os.getcwd(), f'raw_data/{args.dataset}/poses')
    root = video_dir

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    img_dirs = []
    output_files = os.listdir(output_dir)

    processed_videos = []
    for path, subdirs, files in os.walk(output_dir):
        for name in files:
            if name.endswith("_alphapose-results.json"):
                video_name = name.split('_alphapose-results.json')[0]
                processed_videos.append(video_name)
    for path, subdirs, files in os.walk(root):
        for name in files:
            run_pose = False
            # We only care about videos for this use case
            if name.endswith(".mp4") or name.endswith("avi"):
                video_name = name.split('.')[0]
                if video_name in processed_videos:
                    print(f'Skipping {video_name}')
                    continue
                video_filename = os.path.join(path, name)
                video_basename = basename(video_filename)[:-4]
                run_pose = True
            else:
                print(f'Unknown file format: {name}, skipping')
                continue
            if run_pose:
                # Rename constants
                alphapose_orig_results_filename = 'alphapose-results.json'
                alphapose_tracked_results_filename = video_basename + '_alphapose_tracked_person.json'
                alphapose_results_filename = video_basename + '_alphapose-results.json'
                if alphapose_results_filename in output_files:
                    continue
                # Change to AlphaPose dir
                os.chdir(alphapose_dir)

                # Build command line
                command = create_command(alphapose_dir, video_filename, output_dir, is_video=True)
                # # Run command
                print('\n$', command)
                os.system(command)

                # Change back to directory containing this script (main_alpahpose.py)
                os.chdir(output_dir)

                # Convert alphapose-results.json to *_alphapose_tracked_person.json
                read_convert_write(alphapose_orig_results_filename, alphapose_tracked_results_filename)
                # Optionally, rename generic filename 'alphapose-results.json' by adding video filename prefix
                os.rename("alphapose-results.json", alphapose_results_filename)
                shutil.rmtree('poseflow', ignore_errors=True)
                os.chdir(curr_dir)

if __name__ == '__main__':
   main()
