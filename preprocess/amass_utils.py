import os
import numpy as np
import os.path as osp
import torch
from tqdm import tqdm



joints_to_use = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 37
])
joints_to_use = np.arange(0,156).reshape((-1,3))[joints_to_use].reshape(-1)


def read_data(folder, sequences, fps=30, smpl=None, device=torch.device('cpu')):

    theta_dict = {}
    joint_pos_dict = {}

    for seq_name in sequences:
        print(f'Reading {seq_name} sequence...')
        seq_folder = osp.join(folder, seq_name)

        read_sequence(seq_folder, seq_name, theta_dict, joint_pos_dict, fps, smpl, device)

    return theta_dict, joint_pos_dict



def read_sequence(folder, seq_name, theta_dict, joint_pos_dict, fps, smpl, device):
    subjects = list(filter(lambda x: osp.isdir(os.path.join(folder, x)), os.listdir(folder)))

    for subject in tqdm(subjects):
        actions = [x for x in os.listdir(osp.join(folder, subject)) if x.endswith('.npz')]

        for action in actions:
            fname = osp.join(folder, subject, action)
            
            if fname.endswith('shape.npz'):
                continue
                
            data = np.load(fname)
            mocap_framerate = int(data['mocap_framerate'])
            sampling_freq = mocap_framerate // fps
            pose = data['poses'][::sampling_freq, joints_to_use]
            trans = data['trans'][::sampling_freq]

            if pose.shape[0] < 60:
                continue

            shape = np.repeat(data['betas'][:10][np.newaxis], pose.shape[0], axis=0)

            joint_pos = smpl(
                global_orient=torch.zeros((pose.shape[0], 3), device=device, dtype=torch.float32),
                body_pose=torch.tensor(pose[:, 3:], device=device, dtype=torch.float32),
                betas=torch.tensor(shape, device=device, dtype=torch.float32),
                root_trans = torch.zeros((pose.shape[0], 3), device=device, dtype=torch.float32),
                orig_joints=True
            ).joints.cpu().numpy()

            joint_pos_noshape = smpl(
                global_orient=torch.zeros((pose.shape[0], 3), device=device, dtype=torch.float32),
                body_pose=torch.tensor(pose[:, 3:], device=device, dtype=torch.float32),
                betas=torch.zeros((pose.shape[0], 10), device=device, dtype=torch.float32),
                root_trans = torch.zeros((pose.shape[0], 3), device=device, dtype=torch.float32),
                orig_joints=True
            ).joints.cpu().numpy()

            theta = np.concatenate([trans, pose, shape], axis=1)
            vid_name = f'{seq_name}_{subject}_{action[:-4]}'

            theta_dict[vid_name] = theta
            joint_pos_dict[vid_name] = (joint_pos, joint_pos_noshape)

    return
