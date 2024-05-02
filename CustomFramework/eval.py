import random
import numpy as np
import torch
import tqdm
import argparse
from data_transforms.STG import dataset as stg_dataset
from data_transforms.STG.utils import data_utils as stg_data_utils
from models.STG_NF import model_pose as STG_NF_model

def test(model, data, model_args):
    model.eval()
    model.to(model_args['device'])
    pbar = tqdm.tqdm(data)
    probs = torch.empty(0).to(model_args['device'])
    print("Starting Test Eval")
    for itern, data_arr in enumerate(pbar):
        data = [data.to(model_args['device'], non_blocking=True) for data in data_arr]
        score = data[-2].amin(dim=-1)
        samp = data[0][:, :2]
        with torch.no_grad():
            z, nll = model(samp.float(), label=torch.ones(data[0].shape[0]), score=score)
        probs = torch.cat((probs, -1 * nll), dim=0)
    prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
    return prob_mat_np

def main(parser_args):
    # Set seed
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)

    parser_args.vid_path = {
        'train': [],
        'test':  parser_args.vid_path
    }
    parser_args.pose_path = {
        'train': [],
        'test': parser_args.pose_path
    }

    dataset, loader = stg_dataset.get_dataset_and_loader(parser_args, trans_list=stg_data_utils.trans_list, only_test=True)

    model_args = {
        'pose_shape': dataset["test"][0][0][:2].shape,
        'hidden_channels': 0,
        'K': 8,
        'L': 1,
        'R': 3,
        'actnorm_scale': 1.0,
        'flow_permutation': 'permute',
        'flow_coupling': 'affine',
        'LU_decomposed': True,
        'learn_top': False,
        'edge_importance': False,
        'temporal_kernel_size': None,
        'strategy': 'uniform',
        'max_hops': 8,
        'device': 'cuda:0',
    }

    model = STG_NF_model.STG_NF(**model_args)
    STG_NF_model.load_checkpoint(model, parser_args.checkpoint)
    results = test(model, loader['test'], model_args)
    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="STG-NF")
    parser.add_argument('--checkpoint', type=str, metavar='model', help="Path to a pretrained model")
    parser.add_argument('--vid_path', type=str, default=None, help='Path to test vids')
    parser.add_argument('--pose_path', type=str, default=None, help='Path to test pose')
    parser.add_argument('--dataset', type=str, default='ShanghaiTech',
                        choices=['ShanghaiTech', 'ShanghaiTech-HR', 'UBnormal'], help='Dataset for Eval')
    parser_args = parser.parse_args()
    main(parser_args)