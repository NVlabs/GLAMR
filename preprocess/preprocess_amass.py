#!/usr/bin/env python3
"""
Preprocess a consolidated AMASS motions pickle into the dataset format
expected by GLAMR's trajectory prediction CVAE (traj_pred/train.py).

Input:
    --motions_pkl : path to all_motions_fps60.pkl (list of dicts, each with
                    'poses' (T,156), 'trans' (T,3), 'betas' (>=10,),
                    and optionally 'mocap_framerate')

Output (written to --output_path):
    amass_train.pkl       / amass_test.pkl       — theta dicts  {name: (T,85)}
    amass_train_jpos.pkl  / amass_test_jpos.pkl  — joint pos dicts {name: (jpos, jpos_noshape)}

Requires the SMPL body model files at data/body_models/smpl/ (same model
used by traj_pred/train.py at training time for consistency).

Usage:
    cd /path/to/GLAMR
    python preprocess/preprocess_amass.py \
        --motions_pkl /path/to/all_motions_fps60.pkl \
        --output_path datasets/amass_processed/v1 \
        --source_fps 60 \
        --target_fps 30 \
        --train_ratio 0.9
"""

import os
import sys
sys.path.append(os.path.join(os.getcwd()))
import os.path as osp
import pickle
import argparse
import numpy as np
import torch
from amass_utils import read_data_from_pkl
from lib.models.smpl import SMPL, SMPL_MODEL_DIR


parser = argparse.ArgumentParser(description='Preprocess consolidated AMASS pickle for GLAMR traj_pred training')
parser.add_argument('--motions_pkl', required=True,
                    help='Path to consolidated AMASS motions pickle (e.g. all_motions_fps60.pkl)')
parser.add_argument('--output_path', default='datasets/amass_processed/v1',
                    help='Output directory for processed pickle files')
parser.add_argument('--source_fps', type=float, default=60.0,
                    help='Default source FPS (overridden per-sequence if mocap_framerate is present)')
parser.add_argument('--target_fps', type=float, default=30.0,
                    help='Target FPS for output (GLAMR expects 30)')
parser.add_argument('--min_seq_len', type=int, default=60,
                    help='Minimum sequence length after resampling (shorter sequences are dropped)')
parser.add_argument('--train_ratio', type=float, default=0.9,
                    help='Fraction of sequences for training (rest goes to test)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for train/test split')
parser.add_argument('--max_frames_per_batch', type=int, default=2000,
                    help='Max frames per SMPL forward pass (reduce if OOM)')
args = parser.parse_args()


# --- Validate inputs ---
if not osp.isfile(args.motions_pkl):
    raise FileNotFoundError(f'Motions pickle not found: {args.motions_pkl}')

smpl_dir = osp.join(os.getcwd(), SMPL_MODEL_DIR) if not osp.isabs(SMPL_MODEL_DIR) else SMPL_MODEL_DIR
if not osp.isdir(smpl_dir):
    raise FileNotFoundError(
        f'SMPL body model directory not found at {smpl_dir}.\n'
        f'This is required both for preprocessing and for traj_pred training.\n'
        f'Please place SMPL model files (SMPL_NEUTRAL.pkl etc.) in {SMPL_MODEL_DIR}/'
    )

os.makedirs(args.output_path, exist_ok=True)

# --- Load input data ---
print(f'Loading {args.motions_pkl} ...')
with open(args.motions_pkl, 'rb') as f:
    motion_data = pickle.load(f)
print(f'Loaded {len(motion_data)} sequences')

# --- Setup SMPL body model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
smpl = SMPL(SMPL_MODEL_DIR, pose_type='body26fk', create_transl=False).to(device)

# --- Process all sequences ---
theta_dict, joint_pos_dict = read_data_from_pkl(
    motion_data, smpl, device,
    source_fps=args.source_fps,
    target_fps=args.target_fps,
    min_seq_len=args.min_seq_len,
    max_frames_per_batch=args.max_frames_per_batch,
)

print(f'\nProcessed {len(theta_dict)} valid sequences '
      f'(dropped {len(motion_data) - len(theta_dict)} short/invalid)')

if len(theta_dict) == 0:
    print('ERROR: No valid sequences produced. Check input data and min_seq_len.')
    sys.exit(1)

# --- Split into train / test ---
seq_names = sorted(theta_dict.keys())
np.random.seed(args.seed)
indices = np.random.permutation(len(seq_names))
split_idx = int(len(indices) * args.train_ratio)

splits = {
    'train': [seq_names[i] for i in indices[:split_idx]],
    'test':  [seq_names[i] for i in indices[split_idx:]],
}

for split_name, split_seqs in splits.items():
    theta_split = {s: theta_dict[s] for s in split_seqs}
    jpos_split  = {s: joint_pos_dict[s] for s in split_seqs}

    theta_file = osp.join(args.output_path, f'amass_{split_name}.pkl')
    jpos_file  = osp.join(args.output_path, f'amass_{split_name}_jpos.pkl')

    print(f'Saving {split_name} ({len(split_seqs)} seqs) -> {theta_file}')
    with open(theta_file, 'wb') as f:
        pickle.dump(theta_split, f)
    print(f'Saving {split_name} jpos -> {jpos_file}')
    with open(jpos_file, 'wb') as f:
        pickle.dump(jpos_split, f)

# --- Summary ---
total_frames = {s: sum(d.shape[0] for d in split.values())
                for s, split in [('train', {k: theta_dict[k] for k in splits['train']}),
                                 ('test',  {k: theta_dict[k] for k in splits['test']})]}
print(f'\n{"="*60}')
print('PREPROCESSING COMPLETE')
print(f'{"="*60}')
print(f'Output directory : {args.output_path}')
print(f'Target FPS       : {args.target_fps}')
print(f'Train sequences  : {len(splits["train"])}  ({total_frames["train"]} frames)')
print(f'Test sequences   : {len(splits["test"])}  ({total_frames["test"]} frames)')
print(f'\nTo train the trajectory prediction CVAE:')
print(f'  1. Set amass_dir: {args.output_path}  in traj_pred/cfg/traj_pred_demo.yml')
print(f'  2. python traj_pred/train.py --cfg traj_pred_demo --ngpus 1')
