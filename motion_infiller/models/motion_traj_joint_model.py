import time
import torch
import numpy as np
from scipy.interpolate import interp1d
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, print_smpl_joint_val
from lib.utils.torch_transform import angle_axis_to_quaternion
from lib.utils.torch_utils import tensor_to, tensor_to_numpy
from lib.utils.tools import get_eta_str, convert_sec_to_time, find_last_version, get_checkpoint_path
from lib.utils.torch_transform import quat_mul, quat_conjugate, get_heading
from motion_infiller.utils.config import Config as MFillerConfig
from motion_infiller.models import model_dict as mfiller_model_dict
from traj_pred.utils.config import Config as TrajPredConfig
from traj_pred.models import model_dict as traj_model_dict
from traj_pred.utils.traj_utils import traj_local2global_heading


class MotionTrajJointModel:

    def __init__(self, cfg, device=torch.device('cpu'), log=None):
        self.cfg = cfg
        self.specs = self.cfg.model_specs
        self.device = device
        self.log = log
        self.has_recon = True
        self.stochastic = True
        self.multi_step_mfiller = cfg.multi_step_mfiller
        self.multi_step_trajpred = cfg.multi_step_trajpred
        self.smpl = SMPL(SMPL_MODEL_DIR, pose_type='body26fk', create_transl=False).to(device)
        self.load_motion_infiller()
        self.load_trajectory_predictor()

    def load_motion_infiller(self):
        if 'mfiller_cfg' in self.specs:
            self.mfiller_cfg = MFillerConfig(self.specs['mfiller_cfg'], training=False)
            # checkpoint
            if self.mfiller_cfg.model_name != 'mfiller_simple':
                version = self.specs.get('mfiller_version', None)
                version = find_last_version(self.mfiller_cfg.cfg_dir) if version is None else version
                checkpoint_dir = f'{self.mfiller_cfg.cfg_dir}/version_{version}/checkpoints'
                self.mfiller_cp = model_cp = get_checkpoint_path(checkpoint_dir, cp=self.specs.get('mfiller_cp', 'best'))
                if self.log is not None:
                    self.log.info(f'loading motion infiller from check point {model_cp}')
                # model
                self.mfiller = mfiller_model_dict[self.mfiller_cfg.model_name].load_from_checkpoint(model_cp, cfg=self.mfiller_cfg, strict=False)
            else:
                self.mfiller = mfiller_model_dict[self.mfiller_cfg.model_name](self.mfiller_cfg)
            self.mfiller.to(self.device)
            self.mfiller.eval()
            for param in self.mfiller.parameters():
                param.requires_grad_(False)
        else:
            self.mfiller = self.mfiller_cfg = self.mfiller_cp = None

    def load_trajectory_predictor(self):
        if 'trajpred_cfg' in self.specs:
            self.trajpred_cfg = TrajPredConfig(self.specs['trajpred_cfg'], training=False)
            # checkpoint
            version = self.specs.get('trajpred_version', None)
            version = find_last_version(self.trajpred_cfg.cfg_dir) if version is None else version
            checkpoint_dir = f'{self.trajpred_cfg.cfg_dir}/version_{version}/checkpoints'
            self.trajpred_cp = model_cp = get_checkpoint_path(checkpoint_dir, cp=self.specs.get('trajpred_cp', 'best'))
            if self.log is not None:
                self.log.info(f'loading trajectory predictor from check point {model_cp}')
            # model
            self.traj_predictor = traj_model_dict[self.trajpred_cfg.model_name].load_from_checkpoint(model_cp, cfg=self.trajpred_cfg, strict=False)
            self.traj_predictor.to(self.device)
            self.traj_predictor.eval()
            for param in self.traj_predictor.parameters():
                param.requires_grad_(False)
        else:
            self.traj_predictor = self.trajpred_cfg = self.trajpred_cp = None

    def pred_trajectory(self, data, sample_num, recon=False, multi_step=True):
        modes = ['infer', 'recon'] if recon else ['infer']
        for mode in modes:
            if self.traj_predictor.in_joint_pos_only:
                if self.mfiller.model_type == 'angle':
                    assert 'shape' in data
                    body_pose = data[f'{mode}_out_body_pose']
                    shape = data['shape']
                    if mode == 'infer':
                        shape = shape.unsqueeze(1).repeat((1, sample_num, 1, 1))
                    orig_shape = body_pose.shape[:-1]
                    scale = data.get('scale', None)
                    body_pose, shape = body_pose.view(-1, 69), shape.view(-1, 10)
                    joint_pos = self.smpl(
                        body_pose=body_pose,
                        betas=shape,
                        root_trans = torch.zeros_like(body_pose[:, :3]),
                        scale = scale.view(-1) if scale is not None else None,
                        orig_joints=True
                    ).joints
                    pose_key = 'joint_pos'
                    joint_pos = joint_pos[:, 1:, :]
                    motion = joint_pos.reshape(orig_shape + (-1,))
                else:
                    raise NotImplementedError
            else:
                pose_key = 'body_pose' if self.mfiller.model_type == 'angle' else 'joint_pos'
                motion = data[f'{mode}_out_{pose_key}']

            if 'pose' in data:
                data['init_xy'] = data['trans'][:, 0, :2]
                q = angle_axis_to_quaternion(data['pose'][:, 0, :3])
                q = quat_mul(q, quat_conjugate(torch.tensor([0.5, 0.5, 0.5, 0.5], device=q.device)).expand_as(q))
                data['init_heading'] = get_heading(q)

            if mode == 'infer':
                motion = motion.view(-1, *motion.shape[-2:])
                batch = {f'in_{pose_key}': motion}
                if 'in_traj_latent' in data:
                    batch['in_traj_latent'] = data['in_traj_latent']
                if 'init_xy' in data:
                    batch['init_xy'] = data['init_xy'].repeat_interleave(sample_num, dim=0)
                    batch['init_heading'] = data['init_heading'].repeat_interleave(sample_num, dim=0)

                output = self.traj_predictor.inference(batch, sample_num=1, recon=False, multi_step=multi_step)

                if f'{mode}_out_pose' in output:
                    data[f'{mode}_out_pose'] = output[f'{mode}_out_pose'].view(-1, sample_num, *output[f'{mode}_out_pose'].shape[-2:])
                data[f'{mode}_out_trans'] = output[f'{mode}_out_trans'].view(-1, sample_num, *output[f'{mode}_out_trans'].shape[-2:])
                data[f'{mode}_out_orient'] = output[f'{mode}_out_orient'].view(-1, sample_num, *output[f'{mode}_out_orient'].shape[-2:])
                data[f'{mode}_out_local_traj_tp'] = output[f'{mode}_out_local_traj_tp'].view(output[f'{mode}_out_local_traj_tp'].shape[0], -1, sample_num, output[f'{mode}_out_local_traj_tp'].shape[-1])
            else:
                batch = {f'in_{pose_key}': motion, 'pose': data['pose'], 'trans': data['trans'], 'joint_pos_shape': data['joint_pos_shape']}

                output = self.traj_predictor.inference(batch, sample_num=1, recon=True, recon_only=True, multi_step=multi_step)

                if f'{mode}_out_pose' in output:
                    data[f'{mode}_out_pose'] = output[f'{mode}_out_pose']
                data[f'{mode}_out_trans'] = output[f'{mode}_out_trans']
                data[f'{mode}_out_orient'] = output[f'{mode}_out_orient']
                data[f'{mode}_out_local_traj_tp'] = output[f'{mode}_out_local_traj_tp']

    def get_motion_latent(self, seq_len):
        return self.mfiller.get_latent(seq_len)

    def get_traj_latent(self, seq_len):
        return self.traj_predictor.get_latent(seq_len)

    def inference(self, batch, sample_num=5, recon=False):
        data = self.mfiller.inference(batch, sample_num, recon, self.multi_step_mfiller)
        if self.traj_predictor is not None:
            self.pred_trajectory(data, sample_num, recon, self.multi_step_trajpred)
        return data
