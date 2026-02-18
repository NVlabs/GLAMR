import os
import numpy as np
import os.path as osp
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d


# Select 24 joints from SMPL-H's 52-joint layout to match SMPL's 24 joints.
# Joints 0-22 are body joints; joint 37 in SMPL-H = right_index1 = SMPL joint 23.
joints_to_use = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 37
])
joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)


def resample_sequence(data, source_fps, target_fps):
    """Resample a sequence from source_fps to target_fps using linear interpolation."""
    if abs(source_fps - target_fps) < 0.1:
        return data

    T_src = data.shape[0]
    duration = T_src / source_fps
    T_tgt = max(int(round(duration * target_fps)), 2)

    t_src = np.linspace(0, duration, T_src)
    t_tgt = np.linspace(0, duration, T_tgt)

    original_shape = data.shape
    data_flat = data.reshape(T_src, -1)

    interp_func = interp1d(t_src, data_flat, axis=0, kind='linear', fill_value='extrapolate')
    resampled = interp_func(t_tgt)

    return resampled.reshape((T_tgt,) + original_shape[1:])


def compute_joint_positions(smpl, pose_chunk, shape_chunk, device, with_shape=True):
    """
    Compute SMPL 24-joint positions in body-local frame (zero orient, zero trans).

    Args:
        smpl: GLAMR SMPL model instance
        pose_chunk: (T, 72) pose array (24 joints x 3, first 3 = global orient)
        shape_chunk: (T, 10) shape betas
        device: torch device
        with_shape: if True use actual betas, else use zeros

    Returns:
        joints: (T, 24, 3) joint positions
    """
    T = pose_chunk.shape[0]
    betas = torch.tensor(shape_chunk, device=device, dtype=torch.float32) if with_shape \
        else torch.zeros((T, 10), device=device, dtype=torch.float32)

    output = smpl(
        global_orient=torch.zeros((T, 3), device=device, dtype=torch.float32),
        body_pose=torch.tensor(pose_chunk[:, 3:], device=device, dtype=torch.float32),
        betas=betas,
        root_trans=torch.zeros((T, 3), device=device, dtype=torch.float32),
        orig_joints=True
    )
    return output.joints.cpu().numpy()


def read_data_from_pkl(motions_list, smpl, device, source_fps=60.0, target_fps=30.0,
                       min_seq_len=60, max_frames_per_batch=2000):
    """
    Read motion data from a consolidated AMASS pickle (list of dicts).

    Each dict in motions_list should have:
        - 'poses': (T, 156) SMPL-H full pose parameters
        - 'trans': (T, 3) root translation
        - 'betas': (>=10,) shape parameters
        - optionally 'mocap_framerate' to override source_fps per sequence

    Returns:
        theta_dict:     {seq_name: ndarray(T, 85)}  where 85 = trans(3) + pose(72) + shape(10)
        joint_pos_dict: {seq_name: (jpos(T,24,3), jpos_noshape(T,24,3))}
    """
    theta_dict = {}
    joint_pos_dict = {}

    for idx, bdata in enumerate(tqdm(motions_list, desc='Processing sequences')):
        seq_name = f'seq_{idx:06d}'

        poses_full = np.array(bdata['poses'], dtype=np.float64)   # (T, 156)
        trans = np.array(bdata['trans'], dtype=np.float64)         # (T, 3)
        betas = np.array(bdata['betas'], dtype=np.float64)
        betas_10 = betas[:10] if len(betas) >= 10 else np.pad(betas, (0, 10 - len(betas)))

        T = poses_full.shape[0]
        if T < 10:
            continue

        # Per-sequence fps override
        seq_fps = float(bdata.get('mocap_framerate', source_fps))

        # Extract 24-joint pose from SMPL-H 52-joint layout
        pose = poses_full[:, joints_to_use]   # (T, 72)

        # Resample to target fps
        if abs(seq_fps - target_fps) > 0.1:
            pose = resample_sequence(pose, seq_fps, target_fps)
            trans = resample_sequence(trans, seq_fps, target_fps)

        T = pose.shape[0]
        if T < min_seq_len:
            continue

        shape = np.repeat(betas_10[np.newaxis], T, axis=0)  # (T, 10)

        # Compute joint positions in batches
        all_jpos = []
        all_jpos_noshape = []

        for start in range(0, T, max_frames_per_batch):
            end = min(start + max_frames_per_batch, T)
            chunk_pose = pose[start:end].astype(np.float32)
            chunk_shape = shape[start:end].astype(np.float32)

            with torch.no_grad():
                jpos = compute_joint_positions(smpl, chunk_pose, chunk_shape, device, with_shape=True)
                jpos_noshape = compute_joint_positions(smpl, chunk_pose, chunk_shape, device, with_shape=False)

            all_jpos.append(jpos)
            all_jpos_noshape.append(jpos_noshape)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        joint_pos = np.concatenate(all_jpos, axis=0)            # (T, 24, 3)
        joint_pos_noshape = np.concatenate(all_jpos_noshape, axis=0)  # (T, 24, 3)

        theta = np.concatenate([trans, pose, shape], axis=1).astype(np.float32)  # (T, 85)

        theta_dict[seq_name] = theta
        joint_pos_dict[seq_name] = (joint_pos, joint_pos_noshape)

    return theta_dict, joint_pos_dict


# ---------------------------------------------------------------------------
# Legacy API: read from raw AMASS .npz directory structure
# ---------------------------------------------------------------------------

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
