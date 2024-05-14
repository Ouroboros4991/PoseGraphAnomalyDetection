## Setup

To setup this project, I recommend you conda to setup two different environments. This is due to conflicting requirements between STG-NF and TrajREC.
Hence you need to setup 1 environment for STG-NF and another for TrajREC.

To setup these environments, you can follow the steps mentioned in each project:
[Setup STG](https://github.com/orhir/STG-NF/blob/main/README.md) 
[Setup TrajREC](https://github.com/alexandrosstergiou/TrajREC)

To setup AlphaPose, you can follow the steps in the followin link [Setup AlphaPose](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md)
As we make use of the script provided by STG to generated our poses, you should setup AlphaPose in the STG conda environments.

Once the environments are setup, you also need to install the additional requirements specified in requirements.txt

## Running the benchmarks

To run the benchmarks, the following steps need to be executed:
1. Run gen_pose_data.py to extract the poses from the dataset.
2. Run the create_tracks_for_tbdc_rbdc.py script in UBnormal/scripts
3. Run the PrepareUBNormalData notebook to standaridize the format of the ground truth and get the ground truth on pose level
4. Run either the benchmarkSTG.py script or benchmarkTrajREC.py script in their corresponding environment.
5. You can use the the AnalyzeBenchmark notebook to get the results of the benchmark.


## Manually evaluating the models
To manually evaluate the STG model, you can use the following template. You only need to replace the checkpoint with the name of the checkpoint you want to evaluate and update the folders to match your setup.
```
python train_eval.py --dataset UBnormal --seg_len 16  --only_test \
--checkpoint /mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/STG-NF/checkpoints/{checkpoint}.tar \
--pose_path_test '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/STG-NF/data/UBnormal/pose/test' \
--pose_path_train '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/STG-NF/data/UBnormal/pose/train' \
--vid_path_train '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/STG-NF/data/UBnormal/videos' \
--vid_path_test '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/STG-NF/data/UBnormal/videos'
```

Similarly, you can manually evaluate the TrajREC models using the following template:
```
python custom_eval.py \
--trajectories '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/TrajREC/data/UBnormal/training/trajectories' \
--testdata '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/TrajREC/data/UBnormal/testing' \
--batch_size 512 \
--wandb False \
 --eval_only true \
--weights '/mnt/d/VUB/CurrentTrendsOfAI/PoseGraphAnomalyDetection/TrajREC/checkpoints/HRAve_ckpt.pt' \
--video_resolution '1080x720' \
--input_length 3
```

## Statistics
You can calculate the statistics of the UBnormal videos using the script statistics.py in  UBnormal/scripts
In addition, the "Data exploration" notebook provides more insight in the poses themselves.

## References

```bibtex
@inproceedings{stergiou2024holistic,
    title={Holistic Representation Learning for Multitask Trajectory Anomaly Detection},
    author={Stergiou, Alexandros and De Weerdt , Brent and Deligiannis, Nikos},
    booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    year={2024}}

@InProceedings{Hirschorn_2023_ICCV,
    author    = {Hirschorn, Or and Avidan, Shai},
    title     = {Normalizing Flows for Human Pose Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {13545-13554}
}
@InProceedings{Acsintoae_CVPR_2022,
  author    = {Andra Acsintoae and Andrei Florescu and Mariana{-}Iuliana Georgescu and Tudor Mare and  Paul Sumedrea and Radu Tudor Ionescu and Fahad Shahbaz Khan and Mubarak Shah},
  title     = {UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2022},
  }

@article{alphapose,
  author = {Fang, Hao-Shu and Li, Jiefeng and Tang, Hongyang and Xu, Chao and Zhu, Haoyi and Xiu, Yuliang and Li, Yong-Lu and Lu, Cewu},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title = {AlphaPose: Whole-Body Regional Multi-Person Pose Estimation and Tracking in Real-Time},
  year = {2022}
}

@inproceedings{fang2017rmpe,
  title={{RMPE}: Regional Multi-person Pose Estimation},
  author={Fang, Hao-Shu and Xie, Shuqin and Tai, Yu-Wing and Lu, Cewu},
  booktitle={ICCV},
  year={2017}
}

@inproceedings{li2019crowdpose,
    title={Crowdpose: Efficient crowded scenes pose estimation and a new benchmark},
    author={Li, Jiefeng and Wang, Can and Zhu, Hao and Mao, Yihuan and Fang, Hao-Shu and Lu, Cewu},
    booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
    pages={10863--10872},
    year={2019}
}
@inproceedings{xiu2018poseflow,
  author = {Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu},
  title = {{Pose Flow}: Efficient Online Pose Tracking},
  booktitle={BMVC},
  year = {2018}
}
```