import subprocess
import glob
import os
import os.path as osp
import sys
sys.path.append(os.path.join(os.getcwd()))
import pickle
import torch
import argparse
from amass_utils import read_data
from lib.models.smpl import SMPL, SMPL_MODEL_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='datasets/amass')
parser.add_argument('--output_path', default='datasets/amass_processed/v1_new')
parser.add_argument('--video', action='store_true', default=False)
args = parser.parse_args()

amass_dir = args.data_path
processed_dir = args.output_path
untar_files = False
os.makedirs(processed_dir, exist_ok=True)


if untar_files:
    files = sorted(glob.glob(f'{amass_dir}/*.tar.bz2'))
    for fname in files:
        subprocess.call(['tar', '-xvf', fname, '-C', amass_dir])


amass_splits = {
    'train': [
        'BMLhandball',
        'BMLmovi',
        'BioMotionLab_NTroje',
        'CMU',
        'DanceDB',
        'DFaust_67',
        'EKUT',
        'Eyes_Japan_Dataset',
        'KIT',
        'MPI_HDM05',
        'MPI_Limits',
        'MPI_mosh',
        'SFU',
        'TCD_handMocap',
        'TotalCapture'
    ],
    'test': [
        'Transitions_mocap',
        'SSM_synced',
        'HumanEva'
    ]
}


device = torch.device('cuda')
smpl = SMPL(SMPL_MODEL_DIR, pose_type='body26fk', create_transl=False).to(device)

for split in ['test', 'train']:
    theta_data, jpos_data = read_data(amass_dir, sequences=amass_splits[split], smpl=smpl, device=device)

    theta_data_file = osp.join(processed_dir, f'amass_{split}.pkl')
    print(f'Saving AMASS dataset (theta) to {theta_data_file}')
    pickle.dump(theta_data, open(theta_data_file, 'wb'))

    jpos_data_file = osp.join(processed_dir, f'amass_{split}_jpos.pkl')
    print(f'Saving AMASS dataset (joint pos) to {jpos_data_file}')
    pickle.dump(jpos_data, open(jpos_data_file, 'wb'))