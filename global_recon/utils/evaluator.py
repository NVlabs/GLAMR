import os, sys
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from lib.utils.konia_transform import rotation_matrix_to_angle_axis
from lib.utils.torch_transform import batch_compute_similarity_transform_torch, angle_axis_to_quaternion, quat_angle_diff, quaternion_to_angle_axis, quat_mul, quat_apply, inverse_transform, make_transform
from lib.utils.torch_utils import tensor_to
from lib.utils.logging import create_logger
from lib.utils.tools import AverageMeter, concat_lists, find_consecutive_runs
from lib.models.smpl import H36M_TO_J15, SMPL, SMPL_MODEL_DIR, JOINT_REGRESSOR_H36M
from traj_pred.utils.traj_utils import convert_traj_world2heading


def compute_MPJPE(data, mode='all', aligned=False):
    num_data = 0
    mpjpe = 0
    key = 'aligned_eval_joints_world' if aligned else 'eval_joints_world'
    for idx, pose_dict in data['person_data'].items():
        jpos = pose_dict[key]
        gt_jpos = data['gt'][idx][key]
        if mode == 'vis':
            jpos = jpos[pose_dict['vis_frames']]
            gt_jpos = gt_jpos[pose_dict['vis_frames']]
        elif mode == 'invis':
            jpos = jpos[pose_dict['invis_frames']]
            gt_jpos = gt_jpos[pose_dict['invis_frames']]

        if gt_jpos.shape[0] == 0:
            continue

        diff = jpos - gt_jpos
        dist = torch.norm(diff, dim=2)
        mpjpe += dist.mean(dim=1).sum() * 1000
        num_data += diff.shape[0]
    mpjpe = mpjpe / num_data if num_data > 0 else torch.tensor(0)
    info = {'num_data': num_data}
    return mpjpe.item(), info


def compute_PAMPJPE(data, mode='all'):
    num_data = 0
    pampjpe = 0
    for idx, pose_dict in data['person_data'].items():
        jpos = pose_dict['eval_joints_world']
        jpos_pa = pose_dict['eval_joints_world_PA']
        gt_jpos = data['gt'][idx]['eval_joints_world']
        if mode == 'vis':
            jpos = jpos[pose_dict['vis_frames']]
            jpos_pa = jpos_pa[pose_dict['vis_frames']]
            gt_jpos = gt_jpos[pose_dict['vis_frames']]
        elif mode == 'invis':
            jpos = jpos[pose_dict['invis_frames']]
            jpos_pa = jpos_pa[pose_dict['invis_frames']]
            gt_jpos = gt_jpos[pose_dict['invis_frames']]

        if gt_jpos.shape[0] == 0:
            continue

        diff = jpos_pa - gt_jpos
        dist = torch.norm(diff, dim=2)
        pampjpe += dist.mean(dim=1).sum() * 1000
        num_data += diff.shape[0]
    pampjpe = pampjpe / num_data if num_data > 0 else torch.tensor(0)
    info = {'num_data': num_data}
    return pampjpe.item(), info


def compute_PAMPJPE_seq(data, mode='all'):
    num_data = 0
    pampjpe = []
    for idx, pose_dict in data['person_data'].items():
        jpos = pose_dict['eval_joints_world']
        jpos_pa = pose_dict['eval_joints_world_PA']
        gt_jpos = data['gt'][idx]['eval_joints_world']
        if mode == 'vis':
            jpos = jpos[pose_dict['vis_frames']]
            jpos_pa = jpos_pa[pose_dict['vis_frames']]
            gt_jpos = gt_jpos[pose_dict['vis_frames']]
        elif mode == 'invis':
            jpos = jpos[pose_dict['invis_frames']]
            jpos_pa = jpos_pa[pose_dict['invis_frames']]
            gt_jpos = gt_jpos[pose_dict['invis_frames']]

        if gt_jpos.shape[0] == 0:
            pampjpe.append(torch.zeros((0,), device=gt_jpos.device))
            continue

        diff = jpos_pa - gt_jpos
        dist = torch.norm(diff, dim=2)
        pampjpe.append(dist.mean(dim=1) * 1000)
        num_data += diff.shape[0]
    pampjpe = torch.cat(pampjpe).cpu().numpy()
    info = {'num_data': num_data}
    return pampjpe, info


def compute_MPVE(data, mode='all', aligned=False):
    num_data = 0
    mpve = 0
    key = 'aligned_eval_verts_world' if aligned else 'eval_verts_world'
    for idx, pose_dict in data['person_data'].items():
        jpos = pose_dict[key]
        gt_jpos = data['gt'][idx][key]
        if mode == 'vis':
            jpos = jpos[pose_dict['vis_frames']]
            gt_jpos = gt_jpos[pose_dict['vis_frames']]
        elif mode == 'invis':
            jpos = jpos[pose_dict['invis_frames']]
            gt_jpos = gt_jpos[pose_dict['invis_frames']]

        if gt_jpos.shape[0] == 0:
            continue

        diff = jpos - gt_jpos
        dist = torch.norm(diff, dim=2)
        mpve += dist.mean(dim=1).sum() * 1000
        num_data += diff.shape[0]
    mpve = mpve / num_data if num_data > 0 else torch.tensor(0)
    info = {'num_data': num_data}
    return mpve.item(), info


def compute_PAMPJPE_all(data):
    return compute_PAMPJPE(data, 'all')

def compute_PAMPJPE_vis(data):
    return compute_PAMPJPE(data, 'vis')

def compute_PAMPJPE_invis(data):
    return compute_PAMPJPE(data, 'invis')

def compute_sample_PAMPJPE_all(data):
    return compute_PAMPJPE_seq(data, 'all')

def compute_sample_PAMPJPE_vis(data):
    return compute_PAMPJPE_seq(data, 'vis')

def compute_sample_PAMPJPE_invis(data):
    return compute_PAMPJPE_seq(data, 'invis')

def compute_mean_PAMPJPE_invis(data):
    return compute_PAMPJPE_seq(data, 'invis')

def compute_Global_MPJPE(data):
    return compute_MPJPE(data, 'all', aligned=True)

def compute_Global_MPVE(data):
    return compute_MPVE(data, 'all', aligned=True)

def compute_accel_error(data):
    num_data = 0
    accel_err = 0
    for idx, pose_dict in data['person_data'].items():
        jpos = pose_dict['eval_joints_world']
        gt_jpos = data['gt'][idx]['eval_joints_world']
        accel = jpos[:-2] - 2 * jpos[1:-1] + jpos[2:]
        gt_accel = gt_jpos[:-2] - 2 * gt_jpos[1:-1] + gt_jpos[2:]
        diff = accel - gt_accel
        dist = torch.norm(diff, dim=2)
        accel_err += dist.mean(dim=1).sum() * 1000
        num_data += diff.shape[0]
    accel_err /= num_data
    info = {'num_data': num_data}
    return accel_err.item(), info


class Evaluator:

    def __init__(self, algo='', dataset='', device=torch.device('cpu'), log_file='nofile', align_freq=250, compute_sample=True):
        self.algo = algo
        self.dataset = dataset
        self.device = device
        self.align_freq = align_freq
        self.compute_sample = compute_sample
        self.log = create_logger(log_file, file_handle=log_file != 'nofile')
        self.smpl = SMPL(SMPL_MODEL_DIR, pose_type='body26fk', create_transl=False).to(device)
        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float().to(device)
        self.metrics_func = {
            'PA-MPJPE': compute_PAMPJPE_all,
            'PA-MPJPE-vis': compute_PAMPJPE_vis,
            'PA-MPJPE-invis': compute_PAMPJPE_invis,
            'G-MPJPE': compute_Global_MPJPE,
            'G-MPVE': compute_Global_MPVE,
            'ACCEL': compute_accel_error
        }
        self.sample_metrics_func = {
            'sample_PA-MPJPE-invis': compute_sample_PAMPJPE_invis
        }

        if self.compute_sample:
            self.metrics_func.update(self.sample_metrics_func)

        self.metrics_name = list(self.metrics_func.keys())
        self.seed_min_metrics = ['PA-MPJPE-invis']
        self.reset()

    def reset(self):
        self.metrics_dict_collection = dict()
        self.acc_metrics_dict = {'metrics': defaultdict(AverageMeter)}

    def get_aligned_orient_trans(self, pose_dict):
        orient_q = angle_axis_to_quaternion(pose_dict['smpl_orient_world'])
        trans = pose_dict['root_trans_world']
        pose_dict['aligned_orient_q'] = []
        pose_dict['aligned_trans'] = []
        for i in range(int(np.ceil((orient_q.shape[0] / self.align_freq)))):
            sind = i * self.align_freq - int(i > 0)
            eind = min((i + 1) * self.align_freq, orient_q.shape[0])
            aligned_orient_q, aligned_trans = convert_traj_world2heading(orient_q[sind:eind], trans[sind:eind], apply_base_orient_after=True)
            res_start = int(i > 0)
            pose_dict['aligned_orient_q'].append(aligned_orient_q[res_start:])
            pose_dict['aligned_trans'].append(aligned_trans[res_start:])
        pose_dict['aligned_orient_q'] = torch.cat(pose_dict['aligned_orient_q'])
        pose_dict['aligned_orient'] = quaternion_to_angle_axis(pose_dict['aligned_orient_q'])
        pose_dict['aligned_trans'] = torch.cat(pose_dict['aligned_trans'])

    def prepare_seq(self, data):
        
        use_keys = ['pose', 'pose_cam', 'root_trans', 'root_trans_cam', 'smpl_orient_cam', 'smpl_orient_world',\
                    'smpl_pose', 'smpl_beta', 'root_trans_cam', 'root_trans_world', 'scale', 'vis_frames', 'invis_frames', 'visible', 'j3d_h36m', 'kp']
        exclude_keys = ['smpl_pose_rotmat']

        for idx, pose_dict in data['person_data'].items():
            if 'exist_frames' in pose_dict:
                exist_frames = pose_dict['exist_frames']
                gt_dict = data['gt'][idx]
                for key in pose_dict.keys():
                    if not np.any([x in key for x in use_keys]) or key in exclude_keys or pose_dict[key] is None:
                        continue
                    pose_dict[key] = pose_dict[key][exist_frames]
                for key in gt_dict.keys():
                    if not np.any([x in key for x in use_keys]) or key in exclude_keys or gt_dict[key] is None:
                        continue
                    gt_dict[key] = gt_dict[key][exist_frames]

        for coord in ['world']:
            """ GT """
            suffix = '' if coord == 'world' else '_cam'
            for idx, pose_dict in data['gt'].items():
                if f'pose{suffix}' not in pose_dict:
                    continue
                
                visible = data['person_data'][idx]['visible_orig']
                pose_dict['vis_frames'] = visible == 1
                pose_dict['invis_frames'] = visible == 0
                
                pose_dict[f'smpl_orient_{coord}'] = pose_dict[f'pose{suffix}'][:, :3].float()
                pose_dict[f'root_trans_{coord}'] = pose_dict[f'root_trans{suffix}'].float()
                if self.dataset == '3DPW' and coord == 'world':
                    orient_q = angle_axis_to_quaternion(pose_dict[f'smpl_orient_{coord}'])
                    quat = angle_axis_to_quaternion(torch.tensor([np.pi * 0.5, 0, 0], device=self.device)).expand_as(orient_q)
                    pose_dict[f'smpl_orient_{coord}'] = quaternion_to_angle_axis(quat_mul(quat, orient_q))
                    pose_dict[f'root_trans_{coord}'] = quat_apply(quat, pose_dict[f'root_trans_{coord}'])

                smpl_motion = self.smpl(
                    global_orient=pose_dict[f'smpl_orient_{coord}'],
                    body_pose=pose_dict['pose'][:, 3:].float(),
                    betas=pose_dict['shape'].float().repeat(pose_dict['pose'].shape[0], 1),
                    root_trans = pose_dict[f'root_trans_{coord}'],
                    root_scale = None,
                    return_full_pose=True
                )
                pose_dict[f'smpl_verts_{coord}'] = smpl_motion.vertices
                pose_dict[f'smpl_joints_{coord}'] = smpl_motion.joints
                joint_h36m = torch.matmul(self.J_regressor, smpl_motion.vertices)
                joint_15 = joint_h36m[:, H36M_TO_J15]
                pelvis = (joint_15[:, [3]] + joint_15[:, [4]]) * 0.5
                pose_dict[f'eval_joints_{coord}'] = joint_15[:, 1:] - pelvis
                pose_dict[f'eval_verts_{coord}'] = smpl_motion.vertices - pelvis
                
                if coord == 'world':
                    pose_dict['smpl_pose'] = pose_dict['pose'][:, 3:].float()
                    self.get_aligned_orient_trans(pose_dict)
                    # get joints with aligned trajectories
                    smpl_motion = self.smpl(
                        global_orient=pose_dict['aligned_orient'],
                        body_pose=pose_dict['pose'][:, 3:].float(),
                        betas=pose_dict['shape'].float().repeat(pose_dict['pose'].shape[0], 1),
                        root_trans = pose_dict['aligned_trans'],
                        root_scale = None,
                        return_full_pose=True
                    )
                    joint_h36m = torch.matmul(self.J_regressor, smpl_motion.vertices)
                    joint_15 = joint_h36m[:, H36M_TO_J15]
                    pose_dict[f'aligned_eval_joints_{coord}'] = joint_15[:, 1:]
                    pose_dict[f'aligned_eval_verts_{coord}'] = smpl_motion.vertices

            """ Estimation """
            suffix = '_cam' if coord == 'cam' else '_world'
            for idx, pose_dict in data['person_data'].items():

                visible = pose_dict['visible_orig']
                pose_dict['vis_frames'] = visible == 1
                pose_dict['invis_frames'] = visible == 0

                smpl_motion = self.smpl(
                    global_orient=pose_dict[f'smpl_orient{suffix}'],
                    body_pose=pose_dict['smpl_pose'],
                    betas=pose_dict['smpl_beta'],
                    root_trans = pose_dict[f'root_trans{suffix}'],
                    root_scale = pose_dict['scale'] if pose_dict['scale'] is not None else None,
                    return_full_pose=True
                )
                pose_dict[f'smpl_verts_{coord}'] = smpl_motion.vertices
                pose_dict[f'smpl_joints_{coord}'] = smpl_motion.joints
                joint_h36m = torch.matmul(self.J_regressor, smpl_motion.vertices)
                joint_15 = joint_h36m[:, H36M_TO_J15]
                pelvis = (joint_15[:, [3]] + joint_15[:, [4]]) * 0.5
                pose_dict[f'eval_joints_{coord}'] = joint_15[:, 1:] - pelvis
                pose_dict[f'eval_verts_{coord}'] = smpl_motion.vertices - pelvis
                if coord == 'world':
                    self.get_aligned_orient_trans(pose_dict)
                    pose_dict[f'eval_joints_world_PA'] = batch_compute_similarity_transform_torch(pose_dict[f'eval_joints_world'], data['gt'][idx]['eval_joints_world'])         
                    # get joints with aligned trajectories
                    smpl_motion = self.smpl(
                        global_orient=pose_dict['aligned_orient'],
                        body_pose=pose_dict['smpl_pose'],
                        betas=pose_dict['smpl_beta'],
                        root_trans = pose_dict[f'aligned_trans'],
                        root_scale = pose_dict['scale'] if pose_dict['scale'] is not None else None,
                        return_full_pose=True
                    )
                    joint_h36m = torch.matmul(self.J_regressor, smpl_motion.vertices)
                    joint_15 = joint_h36m[:, H36M_TO_J15]
                    pose_dict[f'aligned_eval_joints_{coord}'] = joint_15[:, 1:]
                    pose_dict[f'aligned_eval_verts_{coord}'] = smpl_motion.vertices

    def compute_sequence_metrics(self, data, name=None, accumulate=True):
        data = tensor_to(data, self.device)
        self.prepare_seq(data)
        data['log'] = self.log
        data['name'] = name

        metrics_dict = defaultdict(dict)
        metrics_dict['seq_len'] = data['seq_len']
        for metric, func in self.metrics_func.items():
            val, info = func(data)
            metrics_dict['metrics'][metric] = AverageMeter(val, info['num_data'])

        if accumulate:
            self.update_accumulated_metrics(metrics_dict, name)
        return metrics_dict

    def update_accumulated_metrics(self, metrics_dict, name=None):
        if name is not None:
            self.metrics_dict_collection[name] = metrics_dict
        for metric in self.metrics_name:
            self.acc_metrics_dict['metrics'][metric].update(metrics_dict['metrics'][metric].avg, metrics_dict['metrics'][metric].count)
        return self.acc_metrics_dict

    def metrics_from_multiple_seeds(self, metrics_dict_arr):
        metrics_dict = defaultdict(dict)
        metrics_dict['seq_len'] = metrics_dict_arr[0]['seq_len']
        for metric in self.metrics_name:
            if 'sample' in metric or 'mean' in metric:
                val_arr = np.stack([x['metrics'][metric].avg for x in metrics_dict_arr])
                num_data = metrics_dict_arr[0]['metrics'][metric].count
                if num_data == 0:
                    val = 0
                else:
                    if 'sample' in metric:
                        val = val_arr.min(axis=0)
                    else:
                        val = val_arr.mean(axis=0)
                    val = val.mean()
                metrics_dict['metrics'][metric] = AverageMeter(val, num_data)
            else:
                val_arr = np.array([x['metrics'][metric].avg for x in metrics_dict_arr])
                num_data = metrics_dict_arr[0]['metrics'][metric].count
                if metric in self.seed_min_metrics:
                    val = val_arr.min()
                else:
                    val = val_arr.mean()
                metrics_dict['metrics'][metric] = AverageMeter(val, num_data)
        return metrics_dict
    
    def print_metrics(self, metrics_dict=None, fmt='.3f', prefix='', print_accum=True):
        if metrics_dict is None:
            metrics_dict = self.acc_metrics_dict
        fmt_str = f"%s: %{fmt} (%{fmt})" if print_accum else f"%s: %{fmt}"
        str_stats = f'{prefix}{self.algo} --- ' + ' '.join([fmt_str % ((x, y.avg, y.val) if print_accum else (x, y.avg)) for x, y in metrics_dict['metrics'].items() if not isinstance(y.avg, np.ndarray)])
        if 'sample_PA-MPJPE-invis' not in metrics_dict['metrics']:
            str_stats += ' sample_PA-MPJPE-invis: None (need multiple seeds)'
        self.log.info(str_stats)
