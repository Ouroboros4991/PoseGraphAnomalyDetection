import os
import json
import pandas as pd
import torch
from torch_geometric.data import Dataset, download_url
import numpy as np
from torch_geometric.data import Data
import networkx as nx 
import itertools
import math

from torch_geometric.data import InMemoryDataset

# Ground truth UB normal:
# 1 is normal
# 0 is abnormal

RAW_ORDER = [
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
]
ORDER = []
for s in RAW_ORDER:
    for item in s:
        if isinstance(item, str):
            ORDER.append(item)
            break

def get_files_of_type(directory: str, suffix: str):
    """Function that gets the file names of a certain type in a directory
    """
    filelist = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                json_file = os.path.join(root,file)
                filelist.append(json_file)
    return filelist

def normalize_keypoints(keypoints):
    # Transform 1: Set Nose as origin
    nose_index = ORDER.index('Nose')
    nose_co, _ = keypoints[nose_index]
    tmp_keypoints = []
    for kp in keypoints:
        co, confidence = kp
        adj_x = co[0] - nose_co[0]
        adj_y = co[1] - nose_co[1]
        adj_co = (adj_x, adj_y)
        tmp_keypoints.append((adj_co, confidence))
    # Transofmr 2: Standardize pose size.
    # This is done by setting the distance between the shoulders to 1
    rshoulder_index = ORDER.index('RShoulder')
    lshoulder_index = ORDER.index('LShoulder')
    co1 = keypoints[rshoulder_index][0]
    co2 = keypoints[lshoulder_index][0]
    distance = math.dist(co1, co2)
    normalized_keypoints = []
    for kp in tmp_keypoints:
        co, confidence = kp
        adj_x = co[0]/distance
        adj_y = co[1]/distance
        adj_co = (adj_x, adj_y)
        normalized_keypoints.append((adj_co, confidence))
    return normalized_keypoints


def prepare_tracking_df(json_files, annotation_files_dict):
    """
    Load in the jsons with the corresponding ground truth in the correct format
    """
    dfs = []
    for file in json_files:
        if file.endswith('tracked_person.json'):
            with open(file) as f:
                tracking = json.load(f)
            # The ground truth shared in the repo of STG-NF contains two formats
            # One is a Numpy array, the other is a text file containing per person if which frame contains invalid poses
            # As we want to focus on detecting abnormallies in the pose graph,
            # we only want to have the videos with a grond truth on person basis.
            video = '_'.join(file.split('/')[-1].split('-')[0].split('_')[0:-3])
            try:
                gt_file = annotation_files_dict[video]
                np.load(gt_file)
                continue
            # If there is no ground truth, ignore the video
            except (FileNotFoundError, KeyError):
                continue
            except ValueError:
                with open(gt_file) as file:
                    lines = [line.strip() for line in file]
                gt_video = {}
                for gt in lines:
                    person, start_frame, end_frame = gt.split(',')
                    # Use float to deal with scientific notation
                    person = int(float(person))
                    start_frame = int(float(start_frame))
                    end_frame = int(float(end_frame))
                    gt_video[person] = (start_frame, end_frame)
            items = []
            for person, tracking_data in tracking.items():
                # If it's not in the gt dict, it means that the person has a normal pose
                # for the whole video aka there is no frame in which the person has an abnormal frame
                start_frame, end_frame = gt_video.get(int(person), (-1, -1)) 
                for frame, data in tracking_data.items():
                    frame_number = int(frame)
                    data['frame'] = frame_number
                    if start_frame <= frame_number <= end_frame:
                        data['label'] = 'abnormal'
                    else:
                        data['label'] = 'normal'
                    keypoints = data['keypoints']
                    reformatted_kp = reformat_kp(keypoints)
                    try:
                        normalized_keypoints = normalize_keypoints(reformatted_kp)
                        data['normalized_keypoints'] = normalized_keypoints
                        items.append(data)
                    except ZeroDivisionError:
                        # Skip items with a distance of zero between the shoulders.
                        # This because its likely that the pose is not valid
                        # or is to small to contain discernible features
                        continue
            if items:
                df = pd.DataFrame(items)
                df['video'] = video
                df = df[['video', 'frame', 'label', 'keypoints', 'normalized_keypoints', 'scores']]
                dfs.append(df)
    df_overview = pd.concat(dfs, ignore_index=True)
    return df_overview


def reformat_kp(key_points):
    """Reformat the keypoints so that the array
    contains 1 item per keypoints.
    Each item should have the following format containing
    the coordinates the first element and the confidence as the
    second element:
    Tuple[Tuple[float, float], float]
    """
    co_keypoints = []
    for i in range(0, len(key_points), 3):
        x = float(key_points[i])
        y = float(key_points[i+1])
        c = float(key_points[i+2])
        co = (x,y)
        co_keypoints.append((co, c))
    return co_keypoints


def main():
    raw_ORDER = [
        {0,  "Nose"},
        {1,  "LEye"},
        {2,  "REye"},
        {3,  "LEar"},
        {4,  "REar"},
        {5,  "LShoulder"},
        {6,  "RShoulder"},
        {7,  "LElbow"},
        {8,  "RElbow"},
        {9,  "LWrist"},
        {10, "RWrist"},
        {11, "LHip"},
        {12, "RHip"},
        {13, "LKnee"},
        {14, "Rknee"},
        {15, "LAnkle"},
        {16, "RAnkle"},
    ]
    ORDER = []
    for s in raw_ORDER:
        for item in s:
            if isinstance(item, str):
                ORDER.append(item)
                break
    print('Preparing pose dataset')
    # Read data from general data folder
    poses_folder = os.path.abspath('../raw_data/UBnormal/poses')
    json_files = get_files_of_type(poses_folder, '.json')
    if not json_files:
        raise FileNotFoundError(f'No json files found in the given directory: {poses_folder}')
    annotations_folder = os.path.abspath('../raw_data/UBnormal/annotations')
    annotation_files = get_files_of_type(annotations_folder, '.txt')
    if not annotation_files:
        raise FileNotFoundError(f'No annotation files found in the given directory: {annotations_folder}')
    annotations_dict = {}
    for annotation_file in annotation_files:
        # parse the video name from the full path
        video_name = '_'.join(annotation_file.split('/')[-1].split('-')[0].split('_')[0:-1])
        annotations_dict[video_name] = annotation_file
    df_provided_poses = prepare_tracking_df(json_files, annotations_dict)

    # Split dataset
    print('Splitting dataset')
    df_training = df_provided_poses.sample(frac=0.80)
    df_subset =  df_provided_poses.drop(df_training.index)
    df_validation = df_subset.sample(frac=0.5)
    df_test = df_subset.drop(df_validation.index)

    # Save datasets to csv's in the pygod specific data folder
    print('Saving datasets')
    df_training.to_csv('./data/UBnormal/training.csv', index=False)
    df_validation.to_csv('./data/UBnormal/validation.csv', index=False)
    df_test.to_csv('./data/UBnormal/testing.csv', index=False)

if __name__ == '__main__':
    main()