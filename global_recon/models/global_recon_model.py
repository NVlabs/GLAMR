import time
import os
import torch
import numpy as np
from scipy.interpolate import interp1d
from lib.models.smpl import SMPL, SMPL_MODEL_DIR
from lib.utils.geometry import perspective_projection
from lib.utils.torch_utils import tensor_to, tensor_to_numpy
from lib.utils.tools import get_eta_str, convert_sec_to_time
from lib.utils.joints import get_joints_info
from lib.utils.torch_transform import heading_to_vec, quat_mul, rotation_matrix_to_quaternion, angle_axis_to_quaternion, inverse_transform, make_transform, quat_angle_diff, rot6d_to_rotmat, rotmat_to_rot6d, transform_trans, transform_rot, quaternion_to_angle_axis, vec_to_heading
from global_recon.models.loss_func import loss_func_dict
from motion_infiller.models.motion_traj_joint_model import MotionTrajJointModel
from motion_infiller.utils.config_motion_traj import Config as MotionTrajConfig
from traj_pred.utils.traj_utils import traj_global2local_heading, traj_local2global_heading, interp_orient_q_sep_heading
try:
    from sdf import SDFLoss
except:
    pass


class GlobalReconOptimizer:

    def __init__(self, cfg, device=torch.device('cpu'), log=None):
        self.cfg = cfg
        self.specs = specs = cfg.grecon_model_specs
        self.device = device
        self.log = log

        self.cur_iter = 0
        
        self.smpl = SMPL(SMPL_MODEL_DIR, pose_type='body26fk', create_transl=False).to(device)
        self.use_gt = specs.get('use_gt', False)
        self.est_type = specs.get('est_type', 'hybrik')
        # flags
        self.flag_infer_motion_traj = specs.get('flag_infer_motion_traj', False)
        self.flag_infill_motion = specs.get('flag_infill_motion', True)
        self.flag_pred_traj = specs.get('flag_pred_traj', True)
        self.flag_opt_traj = specs.get('flag_opt_traj', True)
        self.flag_opt_cam = specs.get('flag_opt_cam', True)
        self.flag_fixed_cam = specs.get('flag_fixed_cam', False)
        self.flag_opt_motion_latent = specs.get('flag_opt_motion_latent', False)
        self.flag_opt_traj_latent = specs.get('flag_opt_traj_latent', False)
        self.flag_opt_vis_local_rot = specs.get('flag_opt_vis_local_rot', False)
        self.flag_opt_person2cam_rot = specs.get('flag_opt_person2cam_rot', False)
        self.flag_opt_person2cam_trans = specs.get('flag_opt_person2cam_trans', False)
        self.flag_cam_inv_trans_res_all = specs.get('flag_cam_inv_trans_res_all', True)
        self.flag_filter_pose = specs.get('flag_filter_pose', True)
        self.flag_make_invis_with_keypoint = specs.get('flag_make_invis_with_keypoint', False)
        self.make_invis_keypoint_min_score = specs.get('make_invis_keypoint_min_score', 0.6)
        self.make_invis_keypoint_min_num = specs.get('make_invis_keypoint_min_num', 15)
        self.flag_opt_cam_from_person_pose = specs.get('flag_opt_cam_from_person_pose', False)
        self.flag_init_cam_all_frames = specs.get('flag_init_cam_all_frames', False)
        self.flag_traj_from_cam = specs.get('flag_traj_from_cam', False)
        self.traj_interp_method = specs.get('traj_interp_method', 'linear_interp')
        self.flag_use_pen_loss = specs.get('flag_use_pen_loss', False)
        self.heading_type = specs.get('heading_type', 'scalar')
        self.absolute_heading = specs.get('absolute_heading', False)
        self.cam_fix_frames = specs.get('cam_fix_frames', [[0, None]])
        # main opt
        self.opt_stage_specs = self.cfg.opt_stage_specs

        if self.flag_use_pen_loss:
            self.sdf_loss = SDFLoss(self.smpl.faces, robustifier=True).to(device)

        self.load_model()

    def load_model(self):
        if 'motion_traj_cfg' in self.specs:
            self.mt_cfg = MotionTrajConfig(self.specs['motion_traj_cfg'])
            self.mt_model = MotionTrajJointModel(self.mt_cfg, self.device, self.log)
        else:
            self.mt_cfg = self.mt_model = None

    def init_data(self, in_dict):
        person_data = dict()
        num_fr = len(in_dict['est'][0]['bboxes_dict']['exist'])
        cam_pose = torch.eye(4).repeat((num_fr, 1, 1)).float().to(self.device)
        cam_pose_inv = inverse_transform(cam_pose)
        
        src_joint_info = get_joints_info("smpl")
        dst_joint_info = get_joints_info("body26fk")
        dst_dict = dict((v, k) for k, v in dst_joint_info.name.items())
        smpl_to_body26fk = np.array([(dst_dict[v], k) for k, v in src_joint_info.name.items() if v in dst_dict.keys()])

        if self.est_type == 'hybrik':
            for idx, pose_dict in in_dict['est'].items():
                new_dict = dict()
                new_dict['visible'] = visible = pose_dict['bboxes_dict']['exist'].copy()
                new_dict['visible_orig'] = new_dict['visible'].copy()
                new_dict['fr_start'] = start = np.where(visible)[0][0]
                new_dict['fr_end'] = end = np.where(visible)[0][-1] + 1
                new_dict['exist_frames'] = visible == 1
                new_dict['exist_frames'][start:end] = True
                new_dict['exist_len'] = end - start
                new_dict['max_len'] = max_len = visible.shape[0]
                new_dict['frames'] = np.arange(max_len)
                new_dict['vis_frames'] = vis_frames = visible == 1
                new_dict['invis_frames'] = invis_frames = visible == 0
                new_dict['frame2ind'] = {f: i for i, f in enumerate(new_dict['frames'])}
                new_dict['scale'] = None
                smpl_pose_wroot = quaternion_to_angle_axis(torch.tensor(pose_dict[f'smpl_pose_quat_wroot'], device=self.device)).cpu().numpy()
                new_dict['smpl_pose'] = smpl_pose_wroot[:, 1:].reshape(-1, 69)
                if idx in in_dict['gt']:
                    new_dict['smpl_pose_gt'] = in_dict['gt'][idx]['pose'][:, 3:]
                new_dict['smpl_beta'] = pose_dict['smpl_beta']
                new_dict['smpl_orient_cam'] = smpl_pose_wroot[:, 0]
                new_dict['root_trans_cam'] = pose_dict['root_trans']

                smpl_joints2d = pose_dict['kp_2d'][:, :24]
                kp_2d_with_score = np.zeros((sum(new_dict['vis_frames']), 26, 3))
                smpl_joints2d = np.concatenate((smpl_joints2d, np.ones_like(smpl_joints2d[:, :, [0]])), axis=-1)
                kp_2d_with_score[:, smpl_to_body26fk[:, 0]] = smpl_joints2d[:, smpl_to_body26fk[:, 1]]
                new_dict['kp_2d'] = kp_2d_with_score[:, :, :2]
                new_dict['kp_2d_score'] = kp_2d_with_score[:, :, 2]
                new_dict['kp_2d_aligned'] = new_dict['kp_2d'].copy()
                new_dict['cam_K'] = pose_dict['cam_K'].astype(np.float32)
                # pad motion to have video length
                if not np.all(new_dict['visible']):
                    for key in ['kp_2d', 'kp_2d_score', 'kp_2d_aligned', 'cam_K']:
                        new_val = np.zeros((max_len,) + new_dict[key].shape[1:], dtype=new_dict[key].dtype)
                        new_val[vis_frames] = new_dict[key]
                        new_dict[key] = new_val
                    for key in ['smpl_pose', 'smpl_beta', 'root_trans_cam', 'smpl_orient_cam']:
                        vis_ind = np.where(visible)[0]
                        f = interp1d(vis_ind.astype(np.float32), new_dict[key], axis=0, assume_sorted=True, fill_value="extrapolate")
                        new_val = f(np.arange(max_len, dtype=np.float32))
                        new_dict[key] = new_val
                new_dict = tensor_to(new_dict, self.device)
                if self.flag_filter_pose:
                    self.filter_pose(new_dict)
                # get world pose
                new_dict['root_trans_world'] = transform_trans(cam_pose_inv, new_dict['root_trans_cam'])
                new_dict['smpl_orient_world'] = transform_rot(cam_pose_inv, new_dict['smpl_orient_cam'])
                new_dict['root_trans_world_base'] = new_dict['root_trans_world'].clone()
                new_dict['smpl_orient_world_base'] = new_dict['smpl_orient_world'].clone()
                # mask invisible frames
                new_dict['smpl_pose_nofill'] = new_dict['smpl_pose'].clone()
                new_dict['smpl_pose_nofill'][~new_dict['exist_frames']] = 0.0
                person_data[idx] = new_dict
        else:
            raise ValueError(f'est_type {self.est_type} not supported')

        # perform motion infilling and trajectory prediction
        if self.flag_infer_motion_traj:
            for pose_dict in person_data.values():
                if self.flag_opt_motion_latent:
                    pose_dict['motion_latent'] = self.mt_model.get_motion_latent(seq_len=pose_dict['exist_len']).to(self.device)
                if self.flag_opt_traj_latent:
                    pose_dict['traj_latent'] = self.mt_model.get_traj_latent(seq_len=pose_dict['exist_len']).to(self.device)
                self.infer_motion_traj(pose_dict)
        
        if not (self.flag_infer_motion_traj and self.flag_pred_traj):
            for pose_dict in person_data.values():
                self.init_default_traj(pose_dict)

        # base trans and rot
        for pose_dict in person_data.values():
            pose_dict['person_transform_world'] = make_transform(pose_dict['smpl_orient_world'], pose_dict['root_trans_world'], rot_type='axis_angle')
            pose_dict['person_transform_cam'] = make_transform(pose_dict['smpl_orient_cam'], pose_dict['root_trans_cam'], rot_type='axis_angle')
            pose_dict['person2cam'] = inverse_transform(pose_dict['person_transform_cam'])

        if self.flag_opt_traj:
            for pose_dict in person_data.values():
                if self.flag_opt_person2cam_rot or self.flag_opt_person2cam_trans:
                    pose_dict['person2cam_res_rot'] = torch.tensor([1., 0., 0., 0., 1., 0.], device=pose_dict['person2cam'].device).repeat((pose_dict['person2cam'].shape[0], 1))
                    pose_dict['person2cam_res_trans'] = torch.zeros((pose_dict['person2cam'].shape[0], 3), device=pose_dict['person2cam'].device)
                pose_dict['smpl_orient_world_res'] = torch.zeros_like(new_dict['smpl_orient_world'])
                pose_dict['root_trans_world_res'] = torch.zeros_like(new_dict['root_trans_world'])
            rel_transform_cam = {}
            person_ids = list(person_data.keys())
            for i in range(len(person_ids)):
                for j in range(len(person_ids)):
                    if i != j:
                        rel_transform_cam[(i, j)] = torch.matmul(inverse_transform(person_data[person_ids[i]]['person_transform_cam']), person_data[person_ids[j]]['person_transform_cam'])

            if self.flag_pred_traj:
                for pose_dict in person_data.values():
                    # local traj variables
                    exist_len = pose_dict['exist_len'].sum()
                    pose_dict['traj_local_xy'] = torch.zeros((2,), device=self.device)
                    pose_dict['traj_local_dxy'] = torch.zeros((exist_len - 1, 2), device=self.device)
                    if self.heading_type == 'vec':
                        pose_dict['traj_local_heading'] = torch.zeros((2,), device=self.device)
                        pose_dict['traj_local_dheading'] = torch.zeros((exist_len - 1, 2), device=self.device)
                    else:
                        pose_dict['traj_local_heading'] = torch.zeros((1,), device=self.device)
                        pose_dict['traj_local_dheading'] = torch.zeros((exist_len - 1,), device=self.device)
                        
                    pose_dict['traj_local_z'] = torch.zeros((exist_len,), device=self.device)
                    pose_dict['traj_local_rot'] = torch.zeros((exist_len, 6), device=self.device)
            else:
                for pose_dict in person_data.values():
                    pose_dict['root_trans_world_base'][:] = pose_dict['root_trans_world_base'][0]
                    pose_dict['smpl_orient_world_base'][:] = pose_dict['smpl_orient_world_base'][0]
        else:
            rel_transform_cam = None

        fr_num_persons = sum([pose_dict['vis_frames'] for pose_dict in person_data.values()])

        meta = {
            'algo': 'global_recon', 'mt_cfg': self.mt_cfg.yml_dict, 'num_fr': num_fr
        }

        cam_inv_rot_residual = torch.zeros(((fr_num_persons == 0).sum(), 6)).type_as(cam_pose)
        num_trans = cam_pose.shape[0] if self.flag_cam_inv_trans_res_all else (fr_num_persons == 0).sum()
        cam_inv_trans_residual = torch.zeros((num_trans, 3)).type_as(cam_pose)

        data = {
            'seq_name': in_dict['seq_name'],
            'person_data': person_data,
            'seq_len': cam_pose.shape[0],
            'fr_num_persons': fr_num_persons,
            'cam_pose': cam_pose,
            'cam_pose_inv': cam_pose_inv,
            'cam_inv_rot_residual': cam_inv_rot_residual,
            'cam_inv_trans_residual': cam_inv_trans_residual,
            'rel_transform_cam': rel_transform_cam,
            'gt': in_dict['gt'],
            'gt_meta': in_dict['gt_meta'],
            'meta': meta
        }

        if self.flag_use_pen_loss:
            data['sdf_loss'] = self.sdf_loss

        self.init_cam_pose(data)

        if self.flag_traj_from_cam:
            self.get_traj_from_cam(data)

        if self.flag_infer_motion_traj and self.flag_pred_traj:
            self.init_traj_heading_from_cam(person_data, data)

        if self.flag_init_cam_all_frames:
            self.init_cam_pose(data, all_frames=True)

        self.forward(data, [], {'stage': 'init'})

        return data

    def filter_pose(self, pose_dict):
        visible = pose_dict['visible']
        quat = angle_axis_to_quaternion(pose_dict['smpl_orient_cam'])
        d_angle = quat_angle_diff(quat[1:], quat[:-1])
        angle_threshold = np.pi / 3
        ind = torch.where((d_angle > angle_threshold) & visible[1:].bool())[0] + 1
        for i in ind:
            if visible[i - 1]:
                if i + 1 < quat.shape[0] and visible[i + 1] and i + 1 not in ind:
                    invis_ind = i - 1
                else:
                    invis_ind = i
                visible[invis_ind] = 0
        # keypoint based filtering
        if self.flag_make_invis_with_keypoint:
            vis_ind = torch.where(visible == 1.0)[0]
            scores = pose_dict['kp_2d_score'][vis_ind]
            num_valid_joints = (scores > self.make_invis_keypoint_min_score).sum(dim=1)
            visible[vis_ind[num_valid_joints < self.make_invis_keypoint_min_num]] = 0.0

        pose_dict['vis_frames'] = visible == 1
        pose_dict['invis_frames'] = visible == 0

    def init_traj_heading_from_cam(self, person_data, data):
        for pose_dict in person_data.values():
            pose_in_world = torch.matmul(data['cam_pose_inv'], pose_dict['person_transform_cam'])
            trans = pose_in_world[:, :3, 3]
            orient_q = rotation_matrix_to_quaternion(pose_in_world[:, :3, :3].contiguous())
            orient_q_vis = orient_q[pose_dict['vis_frames']]
            orient_q_interp = interp_orient_q_sep_heading(orient_q_vis, pose_dict['vis_frames'])
            local_rep = traj_global2local_heading(trans, orient_q_interp)
            for (start, end) in self.cam_fix_frames:
                pose_dict['traj_local_pred'][start:end, -2:] = local_rep[pose_dict['exist_frames']][start:end, -2:]
            trans_tp, orient_q_tp = traj_local2global_heading(pose_dict['traj_local_pred'], local_heading=not self.absolute_heading)
            orient_tp = quaternion_to_angle_axis(orient_q_tp)
            exist_fr = pose_dict['exist_frames']
            pose_dict['smpl_orient_world_base'] = pose_dict['smpl_orient_world_base'].detach().clone()
            pose_dict['root_trans_world_base'] = pose_dict['root_trans_world_base'].detach().clone()
            pose_dict['smpl_orient_world_base'][exist_fr] = orient_tp
            pose_dict['root_trans_world_base'][exist_fr] = trans_tp
            pose_dict['smpl_orient_world'] = pose_dict['smpl_orient_world_base'].clone()
            pose_dict['root_trans_world'] = pose_dict['root_trans_world_base'].clone()
            pose_dict['person_transform_world'] = make_transform(pose_dict['smpl_orient_world'], pose_dict['root_trans_world'], rot_type='axis_angle')

    def init_cam_pose(self, data, all_frames=False):
        cam_pose_inv_new = []
        for pose_dict in data['person_data'].values(): 
            cam_pose_inv_new.append(torch.matmul(pose_dict['person_transform_world'], pose_dict['person2cam']) * pose_dict['vis_frames'][:, None, None])
        num_persons = data['fr_num_persons']
        ind = num_persons > 0
        start = torch.where(ind)[0][0]
        data['pose_infer_cam_pose_inv'] = torch.zeros_like(data['cam_pose'])
        data['pose_infer_cam_pose_inv'][ind] = cam_pose_inv_new[0][ind]
        # data['pose_infer_cam_pose_inv'][ind] = sum(cam_pose_inv_new)[ind] / num_persons[ind, None, None]
        if all_frames:
            if not torch.all(ind):
                last_cam = data['pose_infer_cam_pose_inv'][start]
                for i in range(len(num_persons)):
                    if num_persons[i] == 0:
                        data['cam_pose_inv'][i] = last_cam
                    else:
                        last_cam = data['cam_pose_inv'][i]
        else:
            data['pose_infer_cam_pose_inv'][...] = data['pose_infer_cam_pose_inv'][[start]]

        data['pose_infer_cam_pose_inv'][:, :3, :3] = rot6d_to_rotmat(rotmat_to_rot6d(data['pose_infer_cam_pose_inv'][:, :3, :3]))
        data['cam_pose_inv'] = data['pose_infer_cam_pose_inv']
        data['cam_pose'] = inverse_transform(data['cam_pose_inv'])

    def init_default_traj(self, pose_dict):
        pose_dict[f'root_trans_world_base'][:] = torch.tensor([0.0, 0.0, 0.8]).to(self.device)
        pose_dict[f'smpl_orient_world_base'][:] = quaternion_to_angle_axis(torch.tensor([0.0, 0.0, 0.7071, 0.7071]).to(self.device))
        pose_dict[f'root_trans_world'] = pose_dict[f'root_trans_world_base']
        pose_dict[f'smpl_orient_world'] = pose_dict[f'smpl_orient_world_base']

    def get_traj_from_cam(self, data):
        for pose_dict in data['person_data'].values():
            pose_dict['person_transform_world'] = torch.matmul(data['cam_pose_inv'], pose_dict['person_transform_cam'])
            trans = pose_dict['person_transform_world'][:, :3, 3]
            orient_q = rotation_matrix_to_quaternion(pose_dict['person_transform_world'][:, :3, :3].contiguous())

            if self.traj_interp_method == 'linear_interp':
                orient_q_vis = orient_q[pose_dict['vis_frames']]
                orient_q_interp = interp_orient_q_sep_heading(orient_q_vis, pose_dict['vis_frames'])
            elif self.traj_interp_method == 'last_pose':
                last_trans = last_orient_q = last_pose = None
                for fr in torch.where(pose_dict['exist_frames'])[0]:
                    if pose_dict['vis_frames'][fr]:
                        last_trans = trans[fr]
                        last_orient_q = orient_q[fr]
                        last_pose = pose_dict['smpl_pose'][fr]
                    else:
                        trans[fr] = last_trans
                        orient_q[fr] = last_orient_q
                        if not (self.flag_infer_motion_traj and self.flag_infill_motion):
                            pose_dict['smpl_pose'][fr] = last_pose
                    orient_q_interp = orient_q
            else:
                raise ValueError(f'unknown traj interp method: {self.traj_interp_method}!')

            pose_dict[f'root_trans_world'] = pose_dict[f'root_trans_world_base'] = trans
            pose_dict[f'smpl_orient_world'] = pose_dict[f'smpl_orient_world_base'] = quaternion_to_angle_axis(orient_q_interp)

    def infer_motion_traj(self, pose_dict):
        if self.mt_model is not None:
            exist_fr = pose_dict['exist_frames']
            batch = {
                'in_body_pose': pose_dict['smpl_pose_nofill'][exist_fr].unsqueeze(0).clone(),
                'frame_mask': pose_dict['visible'][exist_fr].unsqueeze(0).clone()
            }
            if self.mt_model.traj_predictor is not None and self.mt_model.traj_predictor.in_joint_pos_only:
                batch['shape'] = pose_dict['smpl_beta'][exist_fr].unsqueeze(0)
                batch['scale'] = pose_dict['scale'][exist_fr].unsqueeze(0)
                
            if self.flag_opt_motion_latent:
                batch['in_motion_latent'] = pose_dict['motion_latent']
            if self.flag_opt_traj_latent:
                batch['in_traj_latent'] = pose_dict['traj_latent']
            output = self.mt_model.inference(batch, sample_num=1)

            if self.flag_infill_motion:
                pose_dict['infilled'] = True
                if 'infer_out_body_pose' in output:
                    pose_dict['smpl_pose'] = pose_dict['smpl_pose'].detach().clone()
                    pose_dict['smpl_pose'][exist_fr] = output['infer_out_body_pose'][0, 0]
                if 'infer_out_joint_pos' in output:
                    pose_dict['smpl_joint_pos'] = pose_dict['smpl_joint_pos'].detach().clone()
                    pose_dict['smpl_joint_pos'][exist_fr] = output['infer_out_joint_pos'][0, 0]
                    if 'smpl_pose' in pose_dict:
                        del pose_dict['smpl_pose']

            if self.flag_pred_traj:
                pose_dict['traj_predicted'] = True
                pose_dict['traj_local_pred'] = output['infer_out_local_traj_tp'][:, 0, 0, :].clone()
                pose_dict['smpl_orient_world_base'] = pose_dict['smpl_orient_world_base'].detach().clone()
                pose_dict['root_trans_world_base'] = pose_dict['root_trans_world_base'].detach().clone()
                if 'infer_out_pose' in output:
                    pose_dict['smpl_orient_world_base'][exist_fr] = output['infer_out_pose'][0, 0, :, :3]
                if 'infer_out_orient' in output:
                    pose_dict['smpl_orient_world_base'][exist_fr] = output['infer_out_orient'][0, 0]
                pose_dict['root_trans_world_base'][exist_fr] = output['infer_out_trans'][0, 0]
                pose_dict[f'smpl_orient_world'] = pose_dict[f'smpl_orient_world_base']
                pose_dict[f'root_trans_world'] = pose_dict[f'root_trans_world_base']

    def get_pred_trajectory_base(self, pose_dict, opt_variables):
        exist_fr = pose_dict['exist_frames']
        pose_dict['traj_local'] = pose_dict['traj_local_pred'].detach().clone()
        pose_dict['traj_local'][0, :2] += pose_dict['traj_local_xy']
        pose_dict['traj_local'][1:, :2] += pose_dict['traj_local_dxy']

        dheading_mask = torch.ones_like(pose_dict['traj_local'][1:, 0])
        for (start, end) in self.cam_fix_frames:
            dheading_mask[start:end] = 0.0
        if self.heading_type == 'vec':
            pose_dict['traj_local'][0, -2:] += pose_dict['traj_local_heading']
            pose_dict['traj_local'][1:, -2:] += pose_dict['traj_local_dheading'] * dheading_mask.unsqueeze(1)
        else:
            heading = vec_to_heading(pose_dict['traj_local'][[0], -2:].clone())
            heading += pose_dict['traj_local_heading']
            pose_dict['traj_local'][0, -2:] = heading_to_vec(heading).squeeze(0)

            heading = vec_to_heading(pose_dict['traj_local'][1:, -2:].clone())
            heading += pose_dict['traj_local_dheading'] * dheading_mask
            pose_dict['traj_local'][1:, -2:] = heading_to_vec(heading)

        pose_dict['traj_local'][:, 2] += pose_dict['traj_local_z']
        if self.flag_opt_vis_local_rot:
            pose_dict['traj_local'][pose_dict['vis_frames'], 3:-2] += pose_dict['traj_local_rot'][pose_dict['vis_frames']]
        else:
            pose_dict['traj_local'][:, 3:-2] += pose_dict['traj_local_rot']

        trans_tp, orient_q_tp = traj_local2global_heading(pose_dict['traj_local'], local_heading=not self.absolute_heading)
        orient_tp = quaternion_to_angle_axis(orient_q_tp)
        pose_dict['smpl_orient_world_base'] = pose_dict['smpl_orient_world_base'].detach().clone()
        pose_dict['root_trans_world_base'] = pose_dict['root_trans_world_base'].detach().clone()
        pose_dict['smpl_orient_world_base'][exist_fr] = orient_tp
        pose_dict['root_trans_world_base'][exist_fr] = trans_tp

    def forward(self, data, opt_variables, opt_meta):
        torch.set_printoptions(sci_mode=False)
        
        """ traj and pose computation """
        for pose_dict in data['person_data'].values():
            """ infill motion """
            if self.flag_infer_motion_traj and (self.flag_opt_motion_latent or self.flag_opt_traj_latent):
                opt_latent_start_iter = opt_meta.get('opt_latent_start_iter', 100)
                if self.cur_iter >= opt_latent_start_iter and self.cur_iter % 1 == 0:
                    self.infer_motion_traj(pose_dict)
                else:
                    pose_dict['smpl_pose'] = pose_dict['smpl_pose'].detach()
                    pose_dict['traj_local_pred'] = pose_dict['traj_local_pred'].detach()
                    if self.absolute_heading:
                        d_heading = vec_to_heading(pose_dict['traj_local_pred'][..., -2:])
                        heading = torch.cumsum(d_heading, dim=0)
                        pose_dict['traj_local_pred'][..., -2:] = heading_to_vec(heading)


            """ predict trajectory """
            if self.flag_infer_motion_traj and self.flag_pred_traj:
                self.get_pred_trajectory_base(pose_dict, opt_variables)

            if self.flag_opt_traj:
                if 'world_res' in opt_variables:
                    pose_dict[f'smpl_orient_world'] = pose_dict[f'smpl_orient_world_base'] + pose_dict[f'smpl_orient_world_res']
                    pose_dict[f'root_trans_world'] = pose_dict[f'root_trans_world_base'] + pose_dict[f'root_trans_world_res']
                else:
                    pose_dict[f'smpl_orient_world'] = pose_dict[f'smpl_orient_world_base']
                    pose_dict[f'root_trans_world'] = pose_dict[f'root_trans_world_base']

                if 'world_dheading' in pose_dict:
                    world_dheading = pose_dict['world_dheading']
                    world_dheading_aa = torch.cat((torch.zeros([world_dheading.shape[0], 2], device=self.device), world_dheading), dim=-1)
                    world_dheading_q = angle_axis_to_quaternion(world_dheading_aa)
                    orient_q = quat_mul(world_dheading_q, angle_axis_to_quaternion(pose_dict[f'smpl_orient_world_base']))
                    pose_dict[f'smpl_orient_world'] = quaternion_to_angle_axis(orient_q)
                    pose_dict[f'root_trans_world'] = pose_dict[f'root_trans_world_base']

                if 'world_dxy' in pose_dict:
                    pose_dict[f'root_trans_world'][:, :2] += pose_dict[f'world_dxy']

            pose_dict['person_transform_world'] = make_transform(pose_dict['smpl_orient_world'], pose_dict['root_trans_world'], rot_type='axis_angle')

        """ form camera parameters """
        if self.flag_opt_cam and opt_meta['stage'] != 'init':
            if 'cam' in opt_variables:
                if self.flag_fixed_cam:
                    data['cam_rot_6d'] = data['cam_rot_6d_fix'].expand(data['cam_pose'].shape[0], -1)
                    data['cam_trans'] = data['cam_trans_fix'].expand(data['cam_pose'].shape[0], -1)
                if 'cam_rot_6d' in data:
                    data['cam_pose'] = make_transform(data['cam_rot_6d'], data['cam_trans'], rot_type='6d')
                    data['cam_pose_inv'] = inverse_transform(data['cam_pose'])
            elif self.flag_opt_cam_from_person_pose:
                cam_pose_inv_new = []
                for pose_dict in data['person_data'].values(): 
                    person2cam = pose_dict['person2cam']
                    if self.flag_opt_person2cam_rot or self.flag_opt_person2cam_trans:
                        person2cam_res = make_transform(pose_dict['person2cam_res_rot'], pose_dict['person2cam_res_trans'], rot_type='6d')
                        person2cam = torch.matmul(pose_dict['person2cam'], person2cam_res)
                    cam_pose_inv_new.append(torch.matmul(pose_dict['person_transform_world'], person2cam) * pose_dict['vis_frames'][:, None, None])
                num_persons = data['fr_num_persons']
                ind = num_persons > 0
                data['cam_pose_inv'] = torch.zeros_like(data['cam_pose'])
                data['cam_pose_inv'][ind] = sum(cam_pose_inv_new)[ind] / num_persons[ind, None, None]
                last_cam = data['cam_pose_inv'][torch.where(num_persons > 0)[0][0]]
                for i in range(len(num_persons)):
                    if num_persons[i] == 0:
                        data['cam_pose_inv'][i] = last_cam
                    else:
                        last_cam = data['cam_pose_inv'][i]
                cam_rot_6d = rotmat_to_rot6d(data['cam_pose_inv'][:, :3, :3])
                cam_rot_6d[num_persons == 0] += data['cam_inv_rot_residual']       
                data['cam_pose_inv'][:, :3, :3] = rot6d_to_rotmat(cam_rot_6d)

                if self.flag_cam_inv_trans_res_all:
                    data['cam_pose_inv'][:, :3, 3] += data['cam_inv_trans_residual']
                else:
                    data['cam_pose_inv'][num_persons == 0, :3, 3] += data['cam_inv_trans_residual']
                
                data['cam_pose'] = inverse_transform(data['cam_pose_inv'])
        
        """ mesh and pose related computation """
        for pose_dict in data['person_data'].values():
            pose_dict['smpl_orient_cam_in_world'] = transform_rot(data['cam_pose'], pose_dict['smpl_orient_world'])
            pose_dict['root_trans_cam_in_world'] = transform_trans(data['cam_pose'], pose_dict['root_trans_world'])
            # 2d keypoints
            if 'smpl_pose' in pose_dict and 'cam_K' in pose_dict:
                # only compute loss for visible frames
                smpl_motion = self.smpl(
                    global_orient=pose_dict[f'smpl_orient_world'],
                    body_pose=pose_dict['smpl_pose'],
                    betas=pose_dict['smpl_beta'],
                    root_trans = pose_dict[f'root_trans_world'],
                    root_scale = pose_dict['scale'] if pose_dict['scale'] is not None else None,
                    return_full_pose=True
                )
                joint_3d = smpl_motion.joints
                cam_pose = data['cam_pose']
                joint_3d_cam = transform_trans(cam_pose, joint_3d)
                pose_dict['kp_2d_pred'] = perspective_projection(joint_3d_cam, pose_dict['cam_K'])

                if self.flag_use_pen_loss:
                    pose_dict['smpl_verts'] = smpl_motion.vertices

    def compute_loss(self, data, loss_cfg):
        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        for loss_name in loss_cfg.keys():
            loss_unweighted = loss_func_dict[loss_name](data, loss_cfg[loss_name])
            loss = loss_unweighted * loss_cfg[loss_name]['weight']
            monitor_only = loss_cfg[loss_name].get('monitor_only', False)
            if not monitor_only:
                total_loss += loss
            loss_dict[loss_name] = loss
            loss_unweighted_dict[loss_name] = loss_unweighted
        return total_loss, loss_dict, loss_unweighted_dict

    def optimize_main(self, data, opt_variables, opt_lr, opt_niters, loss_cfg, opt_meta):
        optimizer, param_list = self.init_opt(data, opt_variables, opt_lr)
        loss_uw_dict = None

        def closure():
            nonlocal loss_uw_dict
            optimizer.zero_grad()
            self.forward(data, opt_variables, opt_meta)
            loss, loss_dict, loss_uw_dict = self.compute_loss(data, loss_cfg)
            loss.backward()
            return loss

        for cur_iter in range(opt_niters):
            t_start = time.time()
            self.cur_iter = cur_iter
            if optimizer is not None:
                optimizer.step(closure)
            self.write_logs(loss_uw_dict, meta={'stage': opt_meta['stage'], 't_start': t_start, 'cur_iter': cur_iter, 'opt_niters': opt_niters, 'opt_lr': opt_lr, 'seq_name': data['seq_name']})

        for param in param_list:
            param.requires_grad_(False)
        data['cam_pose'] = data['cam_pose'].detach()
        data['cam_pose_inv'] = data['cam_pose_inv'].detach()
        return data

    def optimize(self, in_dict, continue_opt=False):
        data = tensor_to(in_dict, self.device) if continue_opt else self.init_data(in_dict)

        for stage, stage_specs in self.opt_stage_specs.items():
            opt_variables = stage_specs['opt_variables']
            opt_lr = stage_specs['opt_lr']
            opt_niters = stage_specs['opt_niters']
            loss_cfg = stage_specs['loss_cfg']
            opt_meta = {'stage': stage, 'opt_latent_start_iter': stage_specs.get('opt_latent_start_iter', 0)}

            self.optimize_main(data, opt_variables, opt_lr, opt_niters, loss_cfg, opt_meta)

            if stage_specs.get('reinitialize_cam', False):
                data['cam_pose'][:] = data['cam_pose'][[0]]
                data['cam_pose_inv'] = inverse_transform(data['cam_pose'])

        data = tensor_to_numpy(data)
        return data

    def get_parameter(self, data, opt_variables):
        param_list = []
        if 'cam' not in opt_variables:
            param_list.append(data['cam_inv_rot_residual'])
            param_list.append(data['cam_inv_trans_residual'])
        if 'cam' in opt_variables:
            if self.flag_fixed_cam:
                data['cam_rot_6d_fix'] = rotmat_to_rot6d(data['cam_pose'][[0], :3, :3]).detach()
                data['cam_trans_fix'] = data['cam_pose'][[0], :3, 3].clone().detach()
                param_list.append(data['cam_rot_6d_fix'])
                param_list.append(data['cam_trans_fix'])
            else:
                data['cam_rot_6d'] = rotmat_to_rot6d(data['cam_pose'][:, :3, :3]).detach()
                data['cam_trans'] = data['cam_pose'][:, :3, 3].clone().detach()
                param_list.append(data['cam_rot_6d'])
                param_list.append(data['cam_trans'])
                
        for pose_dict in data['person_data'].values():
            if self.flag_opt_traj:
                for key in opt_variables:
                    if 'world_res' == key:
                        param_list.append(pose_dict['smpl_orient_world_res'])
                        param_list.append(pose_dict['root_trans_world_res'])
                    if 'local' in key:
                        param_list.append(pose_dict[f'traj_{key}'])
            if self.flag_opt_person2cam_rot and 'person2cam_rot' in opt_variables:
                param_list.append(pose_dict['person2cam_res_rot'])
            if self.flag_opt_person2cam_trans and 'person2cam_trans' in opt_variables:
                param_list.append(pose_dict['person2cam_res_trans'])
            if self.flag_opt_motion_latent:
                param_list.append(pose_dict['motion_latent'])
            if self.flag_opt_traj_latent:
                param_list.append(pose_dict['traj_latent'])
            if 'world_dheading' in opt_variables:
                if 'world_dheading' not in pose_dict:
                    pose_dict['world_dheading'] = torch.zeros_like(pose_dict['smpl_orient_world'][..., [0]])
                param_list.append(pose_dict['world_dheading'])
            if 'world_dxy' in opt_variables:
                if 'world_dxy' not in pose_dict:
                    pose_dict['world_dxy'] = torch.zeros_like(pose_dict['smpl_orient_world'][..., :2])
                param_list.append(pose_dict['world_dxy'])

        return param_list

    def init_opt(self, data, opt_variables, opt_lr):
        param_list = self.get_parameter(data, opt_variables)
        for param in param_list:
            param.requires_grad_(True)
        if len(param_list) == 0:
            optimizer = None
        else:
            optimizer = torch.optim.Adam(param_list, lr=opt_lr, betas=(0.9, 0.999))
            # optimizer = torch.optim.LBFGS(param_list, lr=opt_lr, max_iter=1)
        return optimizer, param_list
    
    def write_logs(self, loss_dict, meta):
        metrics_to_ignore = {}
        cur_iter = meta['cur_iter']
        opt_niters = meta['opt_niters']
        opt_lr = meta['opt_lr']
        iter_secs = time.time() - meta['t_start']
        eta_str = get_eta_str(cur_iter, opt_niters, iter_secs)
        loss_str = ' | '.join([f'{x}: {y:7.3f}' for x, y in loss_dict.items() if x not in metrics_to_ignore])
        head_str = f'{self.cfg.id} - {meta["seq_name"]} - {meta["stage"]}'
        info_str = f'{head_str} | {cur_iter:4d}/{opt_niters} | TE: {convert_sec_to_time(iter_secs)} ETA: {eta_str} | LR: {opt_lr:.0e} | {loss_str}'
        if self.log is None:
            print(info_str)
        else:
            self.log.info(info_str)
            