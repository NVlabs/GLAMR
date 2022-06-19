import torch
import pytorch_lightning as pl
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from lib.models import RNN, MLP
from traj_pred.models.loss_func import loss_func_dict
from lib.utils.torch_utils import initialize_weights
from lib.utils.dist import Normal
from traj_pred.utils.traj_utils import traj_global2local_heading, traj_local2global_heading, get_init_heading_q, convert_traj_world2heading, convert_traj_heading2world
from lib.utils.torch_transform import get_heading, get_heading_q, heading_to_vec, quat_mul, quat_apply, quat_conjugate, angle_axis_to_quaternion, quat_to_rot6d, quaternion_to_angle_axis, angle_axis_to_rot6d, rot6d_to_angle_axis, rot6d_to_quat, vec_to_heading
from lib.models.smpl import SMPL, SMPL_MODEL_DIR


""" 
Modules
"""

class ContextEncoder(nn.Module):
    """ Encode context (condition) C, i.e., the input motion, in the CVAE p(X|C) """
    def __init__(self, cfg, specs, ctx, **kwargs):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()
        self.cfg = cfg
        self.specs = specs
        self.ctx = ctx
        self.input_noise = specs.get('input_noise', None)
        self.use_jvel = specs.get('use_jvel', False)
        in_dim = 69
        if self.use_jvel:
            in_dim += 69
        cur_dim = in_dim

        """ in MLP """
        if 'in_mlp' in specs:
            in_mlp_cfg = specs['in_mlp']
            self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.in_mlp.out_dim
        else:
            self.in_mlp = None
        
        """ temporal network """
        temporal_cfg = specs['temporal_net']
        self.t_net_type = temporal_cfg['type']
        num_layers = temporal_cfg.get('num_layers', 1)
        self.temporal_net = nn.ModuleList()
        for _ in range(num_layers):
            net = RNN(cur_dim, temporal_cfg['hdim'], self.t_net_type, bi_dir=temporal_cfg.get('bi_dir', True))
            cur_dim = temporal_cfg['hdim']
            self.temporal_net.append(net)

        """ out MLP """
        if 'out_mlp' in specs:
            out_mlp_cfg = specs['out_mlp']
            self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.out_mlp.out_dim
        else:
            self.out_mlp = None

        if 'context_dim' in specs:
            self.fc = nn.Linear(cur_dim, specs['context_dim'])
            cur_dim = specs['context_dim']
        else:
            self.fc = None
        ctx['context_dim'] = cur_dim

    def forward(self, data):
        x_in = data['in_joint_pos_tp']
        if self.use_jvel:
            x_in = torch.cat([x_in, data['in_joint_vel_tp']], dim=-1)

        if self.training and self.input_noise is not None:
            x_in += torch.randn_like(x_in) * self.input_noise
        
        x = x_in
        if self.in_mlp is not None:
            x = self.in_mlp(x)

        for net in self.temporal_net:
            x = net(x)         

        if self.out_mlp is not None:
            x = self.out_mlp(x)
        if self.fc is not None:
            x = self.fc(x)

        data['context'] = x


class DataEncoder(nn.Module):
    """ Inference (encoder) model q(z|X,C) in CVAE p(X|C) """
    def __init__(self, cfg, specs, ctx, **kwargs):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()
        self.cfg = cfg
        self.specs = specs
        self.ctx = ctx
        self.nz = ctx['nz']    # dimension of latent code z
        self.input = specs.get('input', 'init_heading_coord')
        self.orient_type = specs.get('orient_type', 'axis_angle')
        assert self.orient_type in {'axis_angle', 'quat', '6d'}
        self.pooling = specs['pooling']
        self.append_context = specs['append_context']
        self.use_jvel = specs.get('use_jvel', False)
        if self.input == 'local_traj':
            cur_dim = 11
        else:
            cur_dim = {'axis_angle': 6, 'quat': 7, '6d': 9}[self.orient_type]
        if self.append_context == 'early':
            cur_dim += ctx['context_dim']

        """ in MLP """
        if 'in_mlp' in specs:
            in_mlp_cfg = specs['in_mlp']
            self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.in_mlp.out_dim
        else:
            self.in_mlp = None

        """ temporal network """
        temporal_cfg = specs['temporal_net']
        self.t_net_type = temporal_cfg['type']
        num_layers = temporal_cfg.get('num_layers', 1)
        self.temporal_net = nn.ModuleList()
        for _ in range(num_layers):
            net = RNN(cur_dim, temporal_cfg['hdim'], self.t_net_type, bi_dir=temporal_cfg.get('bi_dir', True))
            cur_dim = temporal_cfg['hdim']
            self.temporal_net.append(net)

        """ out MLP """
        if 'out_mlp' in specs:
            out_mlp_cfg = specs['out_mlp']
            self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.out_mlp.out_dim
        else:
            self.out_mlp = None

        """ fusion MLP """
        if self.append_context == 'late':
            cur_dim += ctx['context_dim']
            fusion_mlp_cfg = specs['fusion_mlp']
            self.fusion_mlp = MLP(cur_dim, fusion_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.fusion_mlp.out_dim
        else:
            self.fusion_mlp = None
        num_dist_params = 2 * self.nz
        self.q_z_net = nn.Linear(cur_dim, num_dist_params)
        initialize_weights(self.q_z_net.modules())

    def forward(self, data):
        context = data['context']
        if self.input == 'global_traj':
            orient_key = {'axis_angle': '', '6d': '_6d', 'quat': '_q_'}[self.orient_type]
            x_in = torch.cat([data['trans_tp'], data[orient_key]], dim=-1)
        elif self.input == 'init_heading_coord':
            init_heading_orient, init_heading_trans = convert_traj_world2heading(data['orient_q_tp'], data['trans_tp'])
            if self.orient_type == 'axis_angle':
                init_heading_orient = quaternion_to_angle_axis(init_heading_orient)
            elif self.orient_type == '6d':
                init_heading_orient = quat_to_rot6d(init_heading_orient)
            x_in = torch.cat([init_heading_trans, init_heading_orient], dim=-1)
        else:
            x_in = data['local_traj_tp'].clone()
            x_in[0, :, [0, 1, -2, -1]] = x_in[1, :, [0, 1, -2, -1]]   # frame 0 stores the abs values of x, y, yaw, copy the relative value from frame 1

        if self.append_context == 'early':
            x_in = torch.cat([x_in, context], dim=-1)

        x = x_in
        if self.in_mlp is not None:
            x = self.in_mlp(x)

        for net in self.temporal_net:
            x = net(x)

        if self.out_mlp is not None:
            x = self.out_mlp(x)

        if self.append_context == 'late':
            x = torch.cat([x, context], dim=-1)
            x = self.fusion_mlp(x)
        
        if self.pooling == 'mean':
            x = x.mean(dim=0)
        else:
            x = x.max(dim=0)

        q_z_params = self.q_z_net(x)
        data['q_z_dist'] = Normal(params=q_z_params)
        data['q_z_samp'] = data['q_z_dist'].rsample()


class DataDecoder(nn.Module):
    """ Likelihood (decoder) model p(X|z,C) in CVAE p(X|C) """
    def __init__(self, cfg, specs, ctx, **kwargs):
        """
        cfg: training cfg file
        specs: module specific specifications
        ctx: context variables shared by all modules, e.g., the output dimension of an upstream module
        """
        super().__init__()
        self.cfg = cfg
        self.specs = specs
        self.ctx = ctx
        self.nz = ctx['nz']    # dimension of latent code z
        self.pooling = specs['pooling']
        self.learn_prior = specs['learn_prior']
        self.deheading_local = ctx['deheading_local']
        self.local_orient_type = ctx['local_orient_type']
        self.use_jvel = specs.get('use_jvel', False)
        self.traj_dim = 11 if self.local_orient_type == '6d' else 8
        cur_dim = ctx['context_dim'] + self.nz

        """ in MLP """
        if 'in_mlp' in specs:
            in_mlp_cfg = specs['in_mlp']
            self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.in_mlp.out_dim
        else:
            self.in_mlp = None
        
        """ temporal network """
        if 'temporal_net' in specs:
            temporal_cfg = specs['temporal_net']
            self.t_net_type = temporal_cfg['type']
            num_layers = temporal_cfg.get('num_layers', 1)
            self.temporal_net = nn.ModuleList()
            for _ in range(num_layers):
                net = RNN(cur_dim, temporal_cfg['hdim'], self.t_net_type, bi_dir=temporal_cfg.get('bi_dir', True))
                cur_dim = temporal_cfg['hdim']
                self.temporal_net.append(net)
        else:
            self.temporal_net = None

        """ out MLP """
        if 'out_mlp' in specs:
            out_mlp_cfg = specs['out_mlp']
            self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.out_mlp.out_dim
        else:
            self.out_mlp = None

        self.out_fc = nn.Linear(cur_dim, self.traj_dim)
        initialize_weights(self.out_fc.modules())

        """ Prior """
        if self.learn_prior:
            cur_dim = ctx['context_dim']
            if 'prior_mlp' in specs:
                prior_mlp_cfg = specs['prior_mlp']
                self.prior_mlp = MLP(cur_dim, prior_mlp_cfg['hdim'], ctx['mlp_htype'])
                cur_dim = self.prior_mlp.out_dim
            else:
                self.prior_mlp = None
            num_dist_params = 2 * self.nz
            self.p_z_net = nn.Linear(cur_dim, num_dist_params)
            initialize_weights(self.p_z_net.modules())


    def forward(self, data, mode, sample_num=1):
        if mode in {'train', 'recon'}:
            assert sample_num == 1
        context = data['context']
        if sample_num > 1:
            context = context.repeat_interleave(sample_num, dim=1)
        # prior p(z) or p(z|C)
        if self.learn_prior:
            if self.pooling == 'mean':
                h = context.mean(dim=0)
            else:
                h = context.max(dim=0)
            h = self.prior_mlp(h)
            p_z_params = self.p_z_net(h)
            data['p_z_dist' + ('_infer' if mode == 'infer' else '')] = Normal(params=p_z_params)
        else:
            data['p_z_dist' + ('_infer' if mode == 'infer' else '')] = Normal(params=torch.zeros(context.shape[1], 2 * self.nz).type_as(context))

        if mode == 'train':
            z = data['q_z_samp']
        elif mode == 'recon':
            z =  data['q_z_dist'].mode()
        elif mode == 'infer':
            eps = data['in_traj_latent'] if 'in_traj_latent' in data else None
            z = data['p_z_dist_infer'].sample(eps)
        else:
            raise ValueError('Unknown Mode!')

        x_in = torch.cat([z.repeat((context.shape[0], 1, 1)), context], dim=-1)
            
        if self.in_mlp is not None:
            x = self.in_mlp(x_in.view(-1, x_in.shape[-1])).view(*x_in.shape[:2], -1)
        else:
            x = x_in

        if self.temporal_net is not None:
            for net in self.temporal_net:
                x = net(x)

        if self.out_mlp is not None:
            x = self.out_mlp(x)
        x = self.out_fc(x)
        x = x.view(-1, data['batch_size'], sample_num, x.shape[-1])
        
        orig_out_local_traj_tp = x
        if mode in {'recon', 'train'}:
             orig_out_local_traj_tp = orig_out_local_traj_tp.squeeze(2)
        data[f'{mode}_orig_out_local_traj_tp'] = orig_out_local_traj_tp

        out_local_traj_tp = x.clone()
        if 'init_xy' in data:
            init_xy = data['init_xy'].unsqueeze(0).unsqueeze(2).repeat((1, 1, sample_num, 1))
            init_heading_vec = heading_to_vec(data['init_heading']).unsqueeze(0).unsqueeze(2).repeat((1, 1, sample_num, 1))
        elif 'local_traj_tp' in data:
            init_xy = data['local_traj_tp'][:1, :, :2].unsqueeze(2).repeat((1, 1, sample_num, 1))
            init_heading_vec = data['local_traj_tp'][:1, :, -2:].unsqueeze(2).repeat((1, 1, sample_num, 1))
        else:
            init_xy = torch.zeros_like(out_local_traj_tp[:1, ..., :2])
            init_heading_vec = torch.tensor([0., 1.], device=init_xy.device).expand_as(out_local_traj_tp[:1, ..., -2:])
        out_local_traj_tp[..., :2] = torch.cat([init_xy, x[1:, ..., :2]], dim=0)   # d_xy
        out_local_traj_tp[..., -2:] = torch.cat([init_heading_vec, x[1:, ..., -2:]], dim=0)   # d_heading_vec
        if mode in {'recon', 'train'}:
             out_local_traj_tp = out_local_traj_tp.squeeze(2)
        data[f'{mode}_out_local_traj_tp'] = out_local_traj_tp
        data[f'{mode}_out_trans_tp'], data[f'{mode}_out_orient_q_tp'] = traj_local2global_heading(out_local_traj_tp, local_orient_type=self.local_orient_type, deheading_local=self.deheading_local)
        # data[f'{mode}_out_orient_6d_tp'] = quat_to_rot6d(data[f'{mode}_out_orient_q_tp'])


""" 
Main Model
"""

class TrajPredVAE(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.model_type = 'joint'
        self.stochastic = True
        self.has_recon = True
        self.cfg = cfg
        """ loss """
        self.loss_cfg = self.cfg.loss_cfg
        self.loss_names = list(self.loss_cfg.keys())
        """ netorks """
        self.setup_networks()
    
    def setup_networks(self):
        self.specs = specs = self.cfg.model_specs
        self.nz = specs['nz']
        self.deheading_local = specs.get('deheading_local', False)
        self.local_orient_type = specs.get('local_orient_type', '6d')
        assert self.local_orient_type in {'6d', 'quat'}
        self.seq_len = self.cfg.seq_len
        self.joint_dropout = specs.get('joint_dropout', 0.0)
        self.joint_from_inpose = specs.get('joint_from_inpose', False)
        self.in_joint_pos_only = specs.get('in_joint_pos_only', False)
        self.ctx = {
            'root_model': self,
            'nz': self.nz,
            'mlp_htype': specs['mlp_htype'],
            'local_orient_type': self.local_orient_type,
            'local_orient_type': self.local_orient_type,
            'deheading_local': self.deheading_local
        }
        self.smpl = SMPL(SMPL_MODEL_DIR, pose_type='body26fk', create_transl=False)
        self.context_encoder = ContextEncoder(self.cfg, specs['context_encoder'], self.ctx)
        self.data_encoder = DataEncoder(self.cfg, specs['data_encoder'], self.ctx)
        self.data_decoder = DataDecoder(self.cfg, specs['data_decoder'], self.ctx)

    def forward(self, data):
        self.context_encoder(data)
        self.data_encoder(data)
        self.data_decoder(data, mode='train')
        return data

    def get_joint_pos(self, body_pose):
        assert not self.in_joint_pos_only
        pose = body_pose.view(-1, 69)
        joints = self.smpl.get_joints(
            global_orient=torch.zeros_like(pose[:, :3]),
            body_pose=pose,
            betas=torch.zeros((pose.shape[0], 10)).type_as(pose),
            root_trans = torch.zeros_like(pose[:, :3]),
        )
        joints = joints[:, 1:, :].reshape(body_pose.shape[:-1] + (-1,))
        return joints

    def init_batch_data(self, batch):
        data = batch.copy()
        # pose
        if 'pose' in data:
            data['pose_tp'] = data['pose'].transpose(0, 1).contiguous()
            data['body_pose_tp'] = data['pose_tp'][..., 3:]
            data['orient_tp'] = data['pose_tp'][..., :3]
            if self.in_joint_pos_only:
                data['joint_pos'] = data['joint_pos_shape']
                data['joint_pos_tp'] = data['joint_pos'].transpose(0, 1).contiguous()
            else:
                data['joint_pos_tp'] = self.get_joint_pos(data['body_pose_tp'])
                data['joint_pos'] = data['joint_pos_tp'].transpose(0, 1).contiguous()
            if self.data_encoder.use_jvel or self.data_decoder.use_jvel:
                data['joint_vel_tp'] = (data['joint_pos_tp'][1:] - data['joint_pos_tp'][:-1]) * 30  # 30 fps
                data['joint_vel_tp'] = torch.cat([data['joint_vel_tp'][[0]], data['joint_vel_tp']], dim=0)

        # in_pose
        if 'in_pose' not in data:
            if 'pose' in data:
                data['in_pose_tp'] = data['pose_tp']
        else:
            data['in_pose_tp'] = data['in_pose'].transpose(0, 1).contiguous()
        # in_body_pose
        if 'in_body_pose' not in data:
            if 'in_pose_tp' in data:
                data['in_body_pose_tp'] = data['in_pose_tp'][..., 3:]
        else:
            data['in_body_pose_tp'] = data['in_body_pose'].transpose(0, 1).contiguous()
        # translation
        if 'trans' in data:
            data['trans_tp'] = data['trans'].transpose(0, 1).contiguous()
            data['orient_q_tp'] = angle_axis_to_quaternion(data['orient_tp'])
            data['orient_6d_tp'] = quat_to_rot6d(data['orient_q_tp'])
            data['local_traj_tp'] = traj_global2local_heading(data['trans_tp'], data['orient_q_tp'], local_orient_type=self.local_orient_type)
        # in_joint_pos
        if 'in_joint_pos' in data:
            data['in_joint_pos_tp'] = data['in_joint_pos'].transpose(0, 1).contiguous()
            if self.context_encoder.use_jvel:
                data['in_joint_vel_tp'] = (data['in_joint_pos_tp'][1:] - data['in_joint_pos_tp'][:-1]) * 30  # 30 fps
                data['in_joint_vel_tp'] = torch.cat([data['in_joint_vel_tp'][[0]], data['in_joint_vel_tp']], dim=0)
        else:
            if 'joint_pos_tp' in data and not self.joint_from_inpose:
                data['in_joint_pos_tp'] = data['joint_pos_tp'].clone()
                if self.context_encoder.use_jvel:
                    data['in_joint_vel_tp'] = data['joint_vel_tp'].clone()
            else:
                data['in_joint_pos_tp'] = self.get_joint_pos(data['in_body_pose_tp'])
                # TODO: joint vel needs special handling for boundary values
                if self.context_encoder.use_jvel:
                    data['in_joint_vel_tp'] = (data['in_joint_pos_tp'][1:] - data['in_joint_pos_tp'][:-1]) * 30  # 30 fps
                    data['in_joint_vel_tp'] = torch.cat([data['in_joint_vel_tp'][[0]], data['in_joint_vel_tp']], dim=0)

        # pose dropout
        if self.training and self.joint_dropout > 0:
            dropout_mask = torch.rand(data['in_joint_pos_tp'].shape[:-1] + (23,), device=data['in_joint_pos_tp'].device)
            dropout_mask = (dropout_mask > self.joint_dropout).float().repeat_interleave(3, dim=-1)
            data['in_joint_pos_tp'] *= dropout_mask

        data['batch_size'] = data['in_joint_pos_tp'].shape[1]
        data['seq_len'] = data['in_joint_pos_tp'].shape[0]
        return data

    def convert_out_pose_trans(self, data, mode, sample_num=1):
        if mode == 'infer':
            data['infer_out_orient_tp'] = quaternion_to_angle_axis(data['infer_out_orient_q_tp'])
            data['infer_out_orient'] = data['infer_out_orient_tp'].permute(1, 2, 0, 3).contiguous()
            data['infer_out_trans'] = data['infer_out_trans_tp'].permute(1, 2, 0, 3).contiguous()
            if 'in_body_pose_tp' in data:
                data['infer_out_pose_tp'] = torch.cat([data['infer_out_orient_tp'], data['in_body_pose_tp'].unsqueeze(2).repeat(1, 1, sample_num, 1)], dim=-1)
                data['infer_out_pose'] = data['infer_out_pose_tp'].permute(1, 2, 0, 3).contiguous()
        else:
            data['recon_out_orient_tp'] = quaternion_to_angle_axis(data['recon_out_orient_q_tp'])
            data['recon_out_orient'] = data['recon_out_orient_tp'].transpose(1, 0).contiguous()
            data['recon_out_trans'] = data['recon_out_trans_tp'].transpose(1, 0).contiguous()
            if 'in_body_pose_tp' in data:
                data['recon_out_pose_tp'] = torch.cat([data['recon_out_orient_tp'], data['in_body_pose_tp']], dim=-1)
                data['recon_out_pose'] = data['recon_out_pose_tp'].transpose(1, 0).contiguous()
    
    def inference_one_step(self, data, sample_num, mode):
        self.context_encoder(data)
        if mode == 'recon':
            self.data_encoder(data)
            self.data_decoder(data, mode=mode)
        else:
            self.data_decoder(data, mode=mode, sample_num=sample_num)
        return data

    def get_seg_data(self, data, sind, eind):
        data_i = dict()
        window_len = eind - sind
        data_i['batch_size'] = data['batch_size']
        data_i['seq_len'] = window_len
        eind_b = min(eind, data['seq_len'])
        data_i['eff_seq_len'] = eind_b - sind
        pad_len = eind - eind_b
        for key in data.keys():
            if 'tp' in key and 'out' not in key:
                data_i[key] = data[key][sind: eind_b].clone()
                if pad_len > 0:
                    pad = torch.zeros((pad_len,) + data[key].shape[1:], device=data[key].device, dtype=data[key].dtype)
                    data_i[key] = torch.cat([data_i[key], pad], dim=0)
        return data_i
    
    def get_res_from_cur_data(self, data, cur_data, sind, eind, mode):
        num_fr = cur_data['eff_seq_len']
        if sind == 0:
            data[f'{mode}_out_local_traj_tp'] = cur_data[f'{mode}_out_local_traj_tp'][:num_fr]
        else:
            cur_data[f'{mode}_orig_out_local_traj_tp'][0, ..., 9:]  = heading_to_vec(get_heading(rot6d_to_quat(data[f'{mode}_out_local_traj_tp'][-1, ..., 3:-2])))
            data[f'{mode}_out_local_traj_tp'] = torch.cat([data[f'{mode}_out_local_traj_tp'], cur_data[f'{mode}_orig_out_local_traj_tp'][:num_fr]], dim=0)

    def inference_multi_step(self, batch, sample_num, recon):
        mode = 'recon' if recon else 'infer'
        data = self.init_batch_data(batch)
        total_len = data['in_joint_pos_tp'].shape[0]
        for i in range(int(np.ceil(total_len / self.seq_len))):
            sind = i * self.seq_len
            eind = (i + 1) * self.seq_len
            data_i = self.get_seg_data(data, sind, eind)
            self.inference_one_step(data_i, sample_num, mode)
            self.get_res_from_cur_data(data, data_i, sind, eind, mode)
        data[f'{mode}_out_trans_tp'], data[f'{mode}_out_orient_q_tp'] = traj_local2global_heading(data[f'{mode}_out_local_traj_tp'], local_orient_type=self.local_orient_type, deheading_local=self.deheading_local)
        return data

    def get_latent(self, seq_len):
        return torch.zeros((1, self.nz))

    def inference(self, batch, sample_num=5, recon=False, recon_only=False, multi_step=False):
        if multi_step:
            if not recon_only:
                data = self.inference_multi_step(batch, sample_num, recon=False)
                self.convert_out_pose_trans(data, 'infer', sample_num)
            if recon:
                data_recon = self.inference_multi_step(batch, sample_num, recon=True)
                if recon_only:
                    data = data_recon
                else:
                    data['recon_out_orient_q_tp'] = data_recon['recon_out_orient_q_tp']
                    data['recon_out_trans_tp'] = data_recon['recon_out_trans_tp']
                    data['recon_out_local_traj_tp'] = data_recon['recon_out_local_traj_tp']
                    self.convert_out_pose_trans(data, 'recon')
        else:
            data = self.init_batch_data(batch)
            self.context_encoder(data)
            if not recon_only:
                self.data_decoder(data, mode='infer', sample_num=sample_num)
                self.convert_out_pose_trans(data, 'infer', sample_num=sample_num)
            if recon:
                self.data_encoder(data)
                self.data_decoder(data, mode='recon')
                self.convert_out_pose_trans(data, 'recon')
        return data

    def decode_only(self, data, z=None, mode='recon', sample_num=1):
        self.data_decoder(data, mode, sample_num=sample_num, z=z)
        self.convert_out_pose_trans(data, mode, sample_num=sample_num)
        return data

    def get_prior(self, batch):
        data = self.init_batch_data(batch)
        self.context_encoder(data)
        prior = self.data_decoder.get_prior(data)
        return prior, data

    def training_step(self, batch, batch_idx):
        data = self.init_batch_data(batch)
        self.forward(data)
        loss, loss_dict, loss_uw_dict = self.compute_loss(data)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        if len(loss_uw_dict) > 1:
            for key, val in loss_uw_dict.items():
                self.log(f'train_{key}', val, on_step=False, on_epoch=True)
        # print('train', data['seq_ind'], data['idx'])
        return loss

    def validation_step(self, batch, batch_idx):
        data = self.init_batch_data(batch)
        self.forward(data)
        loss, loss_dict, loss_uw_dict = self.compute_loss(data)
        self.log('val_loss', loss)
        if len(loss_uw_dict) > 1:
            for key, val in loss_uw_dict.items():
                self.log(f'val_{key}', val, on_step=False, on_epoch=True)
        # print('val', data['seq_ind'], data['idx'])

    def test_step(self, batch, batch_idx):
        data = self.init_batch_data(batch)
        self.forward(data)
        loss, loss_dict, loss_uw_dict = self.compute_loss(data)
        self.log('test_loss', loss)

    """ Loss """
    def compute_loss(self, data):
        total_loss = 0
        loss_dict = {}
        loss_unweighted_dict = {}
        for loss_name in self.loss_names:
            loss_unweighted = loss_func_dict[loss_name](data, self.loss_cfg[loss_name])
            loss = loss_unweighted * self.loss_cfg[loss_name]['weight']
            monitor_only = self.loss_cfg[loss_name].get('monitor_only', False)
            if not monitor_only:
                total_loss += loss
            loss_dict[loss_name] = loss
            loss_unweighted_dict[loss_name] = loss_unweighted
        return total_loss, loss_dict, loss_unweighted_dict

    def on_train_epoch_start(self):
        np.random.seed(self.cfg.seed + 17 * self.current_epoch)
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        np.random.seed(self.cfg.seed + 21 * self.current_epoch)
        return super().on_validation_epoch_start()

    def on_train_epoch_end(self, unused=None):
        self.train_dataloader.dataloader.dataset.epoch_init_seed = None
        return super().on_train_epoch_end(unused=unused)

    def on_validation_epoch_end(self):
        self.val_dataloader.dataloader.dataset.epoch_init_seed = None
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        lr_sched_dict = self.cfg.lr_scheduler
        if lr_sched_dict is None:
            return optimizer
        if lr_sched_dict['type'] == 'reduce_plateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, lr_sched_dict.get('mode', 'min'), factor=lr_sched_dict['factor'], patience=lr_sched_dict.get('patience', 10))
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': lr_sched_dict['monitor']}
        elif lr_sched_dict['type'] == 'step':
            lr_scheduler = StepLR(optimizer, step_size=lr_sched_dict['step_size'], gamma=lr_sched_dict['factor'])
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        else:
            raise ValueError('Unknown lr_scheduler type!')
