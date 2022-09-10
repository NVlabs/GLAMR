import subprocess
import os, sys
sys.path.append(os.path.join(os.getcwd()))
import os.path as osp
import argparse
import torch
import numpy as np
import pickle
import glob

from lib.utils.logging import create_logger
from global_recon.utils.config import Config
from global_recon.models import model_dict
from global_recon.vis.vis_grecon import GReconVisualizer
from pose_est.run_pose_est_demo import run_pose_est_on_video


test_sequences = {
    '3dpw': ['downtown_arguing_00', 'downtown_bar_00', 'downtown_bus_00', 'downtown_cafe_00', 'downtown_car_00', 'downtown_crossStreets_00', 'downtown_downstairs_00', 
             'downtown_enterShop_00', 'downtown_rampAndStairs_00', 'downtown_runForBus_00', 'downtown_runForBus_01', 'downtown_sitOnStairs_00', 'downtown_stairs_00',
             'downtown_upstairs_00', 'downtown_walkBridge_01', 'downtown_walkUphill_00', 'downtown_walking_00', 'downtown_warmWelcome_00', 'downtown_weeklyMarket_00',
             'downtown_windowShopping_00', 'flat_guitar_01', 'flat_packBags_00', 'office_phoneCall_00', 'outdoors_fencing_01'],
    'h36m': list(sorted(glob.glob('datasets/H36M/processed_v1/pose/s_09*.pkl')) + sorted(glob.glob('datasets/H36M/processed_v1/pose/s_11*.pkl')))
}

dataset_paths_dict = {
    '3dpw': {
        'image': 'datasets/3DPW/imageFiles',
        'bbox': 'datasets/3DPW/processed_v1/bbox',
        'gt_pose': 'datasets/3DPW/processed_v1/pose'
    },
    'h36m': {
        'image': 'datasets/H36M/occluded_v2/images',
        'bbox': 'datasets/H36M/occluded_v2/bbox',
        'gt_pose': 'datasets/H36M/occluded_v2/pose'
    }

}


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='3dpw')
parser.add_argument('--cfg', default='glamr_3dpw')
parser.add_argument('--out_dir', default='out/3dpw')
parser.add_argument('--seeds', default="1")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--cached', type=int, default=1)
args = parser.parse_args()


cached = int(args.cached)
cfg = Config(args.cfg, out_dir=args.out_dir)
if torch.cuda.is_available() and args.gpu >= 0:
    device = torch.device('cuda', index=args.gpu)
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device('cpu')

seeds = [int(x) for x in args.seeds.split(',')]
sequences = test_sequences[args.dataset]
dataset_paths = dataset_paths_dict[args.dataset]

# global recon model
grecon_model = model_dict[cfg.grecon_model_name](cfg, device, None)


for i, seq_name in enumerate(sequences[7:]):
    for seed in seeds:
        print(f'{i}/{len(sequences)} seed {seed} processing {seq_name} for {args.dataset}..')
        seq_image_dir = f"{dataset_paths['image']}/{seq_name}"
        seq_out_dir = f"{args.out_dir}/{seq_name}"
        seq_bbox_file = f"{dataset_paths['bbox']}/{seq_name}.pkl"
        seq_gt_pose_file = f"{dataset_paths['gt_pose']}/{seq_name}.pkl"

        cfg.save_yml_file(f'{seq_out_dir}/config.yml')
        grecon_model.log = log = create_logger(f'{cfg.cfg_dir}/log.txt')
        grecon_path = f'{seq_out_dir}/grecon'
        render_path = f'{seq_out_dir}/grecon_videos'
        os.makedirs(grecon_path, exist_ok=True)
        os.makedirs(render_path, exist_ok=True)

        pose_est_dir = f'{seq_out_dir}/pose_est'
        log.info(f"running {cfg.grecon_model_specs['est_type']} pose estimation on {seq_image_dir}...")
        run_pose_est_on_video(None, pose_est_dir, cfg.grecon_model_specs['est_type'], image_dir=seq_image_dir, bbox_file=seq_bbox_file, cached_pose=cached, gpu_index=args.gpu)
        pose_est_model_name = {'hybrik': 'HybrIK'}[cfg.grecon_model_specs['est_type']]

        np.random.seed(seed)
        torch.manual_seed(seed)

        pose_est_file = f'{pose_est_dir}/pose.pkl'
        log.info(f'running global reconstruction on {seq_image_dir}, seed: {seed}')
        seq_name = osp.basename(seq_image_dir)

        # main
        out_file = f'{grecon_path}/{seq_name}_seed{seed}.pkl'

        est_dict = pickle.load(open(pose_est_file, 'rb'))
        if seq_gt_pose_file is None:
            in_dict = {'est': est_dict, 'gt': dict(), 'gt_meta': dict(), 'seq_name': seq_name}
        else:
            gt_dict = pickle.load(open(seq_gt_pose_file, 'rb'))
            in_dict = {'est': est_dict, 'gt': gt_dict['person_data'], 'gt_meta': gt_dict['meta'], 'seq_name': seq_name}
        # global optimization
        out_dict = grecon_model.optimize(in_dict)
        pickle.dump(out_dict, open(out_file, 'wb'))

