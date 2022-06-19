import torch
import pytorch_lightning as pl
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from lib.models import RNN, MLP, PositionalEncoding
from motion_infiller.models.loss_func import loss_func_dict
from traj_pred.utils.config import Config as TrajPredConfig
from traj_pred.models import model_dict as traj_model_dict
from lib.utils.torch_utils import initialize_weights, ExtModuleWrapper
from lib.utils.dist import Normal
from lib.utils.torch_transform import angle_axis_to_rot6d, rot6d_to_angle_axis
from lib.utils.tools import find_last_version, get_checkpoint_path
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
        self.past_nframe = ctx['past_nframe']
        self.cur_nframe = ctx['cur_nframe']
        self.fut_nframe = ctx['fut_nframe']
        self.input_noise = specs.get('input_noise', None)
        self.use_jpos = specs.get('use_jpos', False)
        self.use_jvel = specs.get('use_jvel', False)
        self.pose_rep = ctx['pose_rep']
        self.rot_type = specs.get('rot_type', 'axis_angle')
        assert self.rot_type in {'axis_angle', '6d'}
        pose_dim = (69 if self.pose_rep == 'body' else 72) * (2 if self.rot_type == '6d' else 1)
        if self.use_jpos:
            pose_dim += 69
        if self.use_jvel:
            pose_dim += 69
        cur_dim = pose_dim

        """ in MLP """
        if 'in_mlp' in specs:
            in_mlp_cfg = specs['in_mlp']
            self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.in_mlp.out_dim
        else:
            self.in_mlp = None
        
        if 'in_fc' in specs:
            self.in_fc = nn.Linear(cur_dim, specs['in_fc'])
            cur_dim = specs['in_fc']
        else:
            self.in_fc = None

        """ temporal network """
        temporal_cfg = specs['transformer']
        # positional encoding
        pe_cfg = specs['transformer']['positional_encoding']
        max_freq = pe_cfg.get('max_freq', 10)
        freq_scale = pe_cfg.get('freq_scale', 0.1)
        concat = pe_cfg.get('concat', True)
        self.pos_enc = PositionalEncoding(temporal_cfg['model_dim'], cur_dim, pe_cfg['enc_type'], max_freq, freq_scale, concat=concat)
        # transformer
        tf_layers = nn.TransformerEncoderLayer(temporal_cfg['model_dim'], temporal_cfg['nhead'], temporal_cfg['ff_dim'], temporal_cfg['dropout'])
        self.temporal_net = nn.TransformerEncoder(tf_layers, temporal_cfg['nlayer'])
        self.temporal_dim = cur_dim = temporal_cfg['model_dim']

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
        x_in = data['in_body_pose_tp'] if self.pose_rep == 'body' else data['in_pose_tp']
        if self.rot_type == '6d':
            aa = x_in.view(x_in.shape[:-1] + (-1, 3))
            sixd = angle_axis_to_rot6d(aa)
            x_in = sixd.view(x_in.shape[:-1] + (-1,))

        if self.use_jpos:
            x_in = torch.cat([x_in, data['in_joint_pos_tp']], dim=-1)
        if self.use_jvel:
            x_in = torch.cat([x_in, data['in_joint_vel_tp']], dim=-1)

        data['x_in'] = x_in

        if self.training and self.input_noise is not None:
            x_in += torch.randn_like(x_in) * self.input_noise
        
        x = x_in
        if self.in_mlp is not None:
            x = self.in_mlp(x)
        if self.in_fc is not None:
            x = self.in_fc(x)

        x = self.pos_enc(x)
        x = self.temporal_net(x, src_key_padding_mask=data['vis_frame_mask'])             

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
        self.past_nframe = ctx['past_nframe']
        self.cur_nframe = ctx['cur_nframe']
        self.fut_nframe = ctx['fut_nframe']
        self.pooling = specs['pooling']
        self.sep_vis_head = specs.get('sep_vis_head', False)
        self.masked_pose_only = specs.get('masked_pose_only', False)
        self.use_jpos = specs.get('use_jpos', False)
        self.use_jvel = specs.get('use_jvel', False)
        self.pose_rep = ctx['pose_rep']
        self.rot_type = specs.get('rot_type', 'axis_angle')
        assert self.rot_type in {'axis_angle', '6d'}
        pose_dim = (69 if self.pose_rep == 'body' else 72) * (2 if self.rot_type == '6d' else 1)
        if self.use_jpos:
            pose_dim += 69
        if self.use_jvel:
            pose_dim += 69
        cur_dim = pose_dim
    
        """ in MLP """
        if 'in_mlp' in specs:
            in_mlp_cfg = specs['in_mlp']
            self.in_mlp = MLP(cur_dim, in_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.in_mlp.out_dim
        else:
            self.in_mlp = None

        """ temporal network """
        temporal_cfg = specs['transformer']
        if cur_dim != temporal_cfg['model_dim']:
            self.in_fc = nn.Linear(cur_dim, temporal_cfg['model_dim'])
            cur_dim = temporal_cfg['model_dim']
        else:
            self.in_fc = None
        # positional encoding
        pe_cfg = specs['transformer']['positional_encoding']
        max_freq = pe_cfg.get('max_freq', 10)
        freq_scale = pe_cfg.get('freq_scale', 0.1)
        concat = pe_cfg.get('concat', True)
        learnable_pos_index = pe_cfg.get('learnable_pos_index', None)
        self.pos_enc = PositionalEncoding(temporal_cfg['model_dim'], cur_dim, pe_cfg['enc_type'], max_freq, freq_scale, concat=concat, learnable_pos_index=learnable_pos_index)
        # transformer
        tf_layers = nn.TransformerDecoderLayer(temporal_cfg['model_dim'], temporal_cfg['nhead'], temporal_cfg['ff_dim'], temporal_cfg['dropout'])
        self.temporal_net = nn.TransformerDecoder(tf_layers, temporal_cfg['nlayer'])
        self.temporal_dim = cur_dim = temporal_cfg['model_dim']

        """ out MLP """
        if 'out_mlp' in specs:
            out_mlp_cfg = specs['out_mlp']
            self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.out_mlp.out_dim
        else:
            self.out_mlp = None

        if self.pooling == 'attn':
            self.mu_token = nn.Parameter(torch.randn(temporal_cfg['model_dim']) * 0.01)
            self.logvar_token = nn.Parameter(torch.randn(temporal_cfg['model_dim']) * 0.01)
            self.q_z_mu_net = nn.Linear(cur_dim, self.nz)
            self.q_z_logvar_net = nn.Linear(cur_dim, self.nz)
            initialize_weights(self.q_z_mu_net.modules())
            initialize_weights(self.q_z_logvar_net.modules())
        else:
            num_dist_params = 2 * self.nz
            self.q_z_net = nn.Linear(cur_dim, num_dist_params)
            initialize_weights(self.q_z_net.modules())

    def forward(self, data):
        context = data['context']
        x_in = data['body_pose_tp'] if self.pose_rep == 'body' else data['pose_tp']
        x_in = x_in[self.past_nframe:-self.fut_nframe]
        if self.rot_type == '6d':
            aa = x_in.view(x_in.shape[:-1] + (-1, 3))
            sixd = angle_axis_to_rot6d(aa)
            x_in = sixd.view(x_in.shape[:-1] + (-1,))

        if self.masked_pose_only:
            x_in *= 1 - data['pose_mask_tp'][self.past_nframe:-self.fut_nframe]

        if self.use_jpos:
            x_in = torch.cat([x_in, data['joint_pos_tp'][self.past_nframe:-self.fut_nframe]], dim=-1)
        if self.use_jvel:
            x_in = torch.cat([x_in, data['joint_vel_tp'][self.past_nframe:-self.fut_nframe]], dim=-1)

        x = x_in
        if self.in_mlp is not None:
            x = self.in_mlp(x)
        if self.in_fc is not None:
            x = self.in_fc(x)

        if self.pooling == 'attn':
            x = torch.cat([self.mu_token.repeat(1, x.shape[1], 1), self.logvar_token.repeat(1, x.shape[1], 1), x], dim=0)
            x = self.pos_enc(x)
            x = self.temporal_net(x, context, memory_key_padding_mask=data['vis_frame_mask'])
            mu = self.q_z_mu_net(x[0])
            logvar = self.q_z_logvar_net(x[1])
            data['q_z_dist'] = Normal(mu=mu, logvar=logvar)
        else:
            x = self.pos_enc(x)
            x = self.temporal_net(x, context, memory_key_padding_mask=data['vis_frame_mask'])

            if self.out_mlp is not None:
                x = self.out_mlp(x)

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
        self.past_nframe = ctx['past_nframe']
        self.cur_nframe = ctx['cur_nframe']
        self.fut_nframe = ctx['fut_nframe']
        self.use_pos_offset = specs.get('use_pos_offset', True)
        self.pooling = specs['pooling']
        self.learn_prior = specs['learn_prior']
        self.pred_past = specs.get('pred_past', False)
        self.use_jpos = specs.get('use_jpos', False)
        self.use_jvel = specs.get('use_jvel', False)
        self.pose_rep = ctx['pose_rep']
        self.rot_type = specs.get('rot_type', 'axis_angle')
        assert self.rot_type in {'axis_angle', '6d'}
        pose_dim = (69 if self.pose_rep == 'body' else 72) * (2 if self.rot_type == '6d' else 1)
        if self.use_jpos:
            pose_dim += 69
        if self.use_jvel:
            pose_dim += 69
        cur_dim = ctx['context_dim']

        """ temporal network """
        temporal_cfg = specs['transformer']
        if cur_dim != temporal_cfg['model_dim']:
            self.in_fc = nn.Linear(cur_dim, temporal_cfg['model_dim'])
            cur_dim = temporal_cfg['model_dim']
        else:
            self.in_fc = None
        # positional encoding
        pe_cfg = specs['transformer']['positional_encoding']
        max_freq = pe_cfg.get('max_freq', 10)
        freq_scale = pe_cfg.get('freq_scale', 0.1)
        concat = pe_cfg.get('concat', True)
        self.pos_enc = PositionalEncoding(temporal_cfg['model_dim'], self.nz, pe_cfg['enc_type'], max_freq, freq_scale, concat=concat)
        # transformer
        tf_layers = nn.TransformerDecoderLayer(temporal_cfg['model_dim'], temporal_cfg['nhead'], temporal_cfg['ff_dim'], temporal_cfg['dropout'])
        self.temporal_net = nn.TransformerDecoder(tf_layers, temporal_cfg['nlayer'])
        self.temporal_dim = cur_dim = temporal_cfg['model_dim']

        """ out MLP """
        if 'out_mlp' in specs:
            out_mlp_cfg = specs['out_mlp']
            self.out_mlp = MLP(cur_dim, out_mlp_cfg['hdim'], ctx['mlp_htype'])
            cur_dim = self.out_mlp.out_dim
        else:
            self.out_mlp = None

        self.out_fc = nn.Linear(cur_dim, pose_dim)
        initialize_weights(self.out_fc.modules())

        """ Prior """
        if self.learn_prior:
            cur_dim = ctx['context_dim']
            if self.pooling == 'attn':
                temporal_cfg = specs['prior_transformer']
                pe_cfg = specs['prior_transformer']['positional_encoding']
                max_freq = pe_cfg.get('max_freq', 10)
                freq_scale = pe_cfg.get('freq_scale', 0.1)
                concat = pe_cfg.get('concat', True)
                learnable_pos_index = pe_cfg.get('learnable_pos_index', None)
                self.prior_pos_enc = PositionalEncoding(temporal_cfg['model_dim'], cur_dim, pe_cfg['enc_type'], max_freq, freq_scale, concat=concat, learnable_pos_index=learnable_pos_index)
                tf_layers = nn.TransformerDecoderLayer(temporal_cfg['model_dim'], temporal_cfg['nhead'], temporal_cfg['ff_dim'], temporal_cfg['dropout'])
                self.prior_temporal_net = nn.TransformerDecoder(tf_layers, temporal_cfg['nlayer'])
                self.prior_temporal_dim = cur_dim = temporal_cfg['model_dim']
                self.mu_token = nn.Parameter(torch.randn(temporal_cfg['model_dim']) * 0.01)
                self.logvar_token = nn.Parameter(torch.randn(temporal_cfg['model_dim']) * 0.01)
                self.p_z_mu_net = nn.Linear(cur_dim, self.nz)
                self.p_z_logvar_net = nn.Linear(cur_dim, self.nz)
                initialize_weights(self.p_z_mu_net.modules())
                initialize_weights(self.p_z_logvar_net.modules())
            else:
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
        mask = data['vis_frame_mask']
        if sample_num > 1:
            context = context.repeat_interleave(sample_num, dim=1)
            mask = mask.repeat_interleave(sample_num, dim=0)
        # prior p(z) or p(z|C)
        if self.learn_prior:
            if self.pooling == 'attn':
                x = torch.cat([self.mu_token.repeat(1, context.shape[1], 1), self.logvar_token.repeat(1, context.shape[1], 1)], dim=0)
                x = self.prior_pos_enc(x)
                x = self.prior_temporal_net(x, context, memory_key_padding_mask=mask)     # TODO: fix masking
                mu = self.p_z_mu_net(x[0])
                logvar = self.p_z_logvar_net(x[1])
                data['p_z_dist' + ('_infer' if mode == 'infer' else '')] = Normal(mu=mu, logvar=logvar)
            else:
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
            eps = data['in_motion_latent'] if 'in_motion_latent' in data else None
            z = data['p_z_dist_infer'].sample(eps)
        else:
            raise ValueError('Unknown Mode!')

        z_in = z.repeat((self.cur_nframe + self.past_nframe if self.pred_past else self.cur_nframe, 1, 1))
            
        x_ctx = context
        if self.in_fc is not None:
            x_ctx = self.in_fc(x_ctx)

        use_pos_offset = False if self.pred_past else self.use_pos_offset
        pe = self.pos_enc(z_in, pos_offset=self.past_nframe if use_pos_offset else 0)
        x = self.temporal_net(pe, x_ctx, memory_key_padding_mask=mask)

        if self.out_mlp is not None:
            x = self.out_mlp(x)
        x = self.out_fc(x)

        if not self.pred_past:
            x = torch.cat([data['x_in'][:self.past_nframe].repeat_interleave(sample_num, dim=1), x], dim=0)
            
        x_all = x.view(-1, data['batch_size'], sample_num, x.shape[-1])

        x = x_all[..., :69] # pose only

        if mode in {'recon', 'train'}:
             x = x.squeeze(2)

        if self.rot_type == '6d':
            data[f'{mode}_out_body_pose_6d_tp'] = x
            sixd = x.view(x.shape[:-1] + (-1, 6))
            aa = rot6d_to_angle_axis(sixd)
            x = aa.reshape(x.shape[:-1] + (-1,))
             
        if self.pose_rep == 'body':
            data[f'{mode}_out_body_pose_tp'] = x
            root_rot = data['pose_tp'][:-self.fut_nframe, :, :3] if 'pose_tp' in data else torch.zeros_like(data['in_body_pose_tp'][:-self.fut_nframe, :, :3])
            if mode == 'infer':
                root_rot = root_rot.unsqueeze(2).repeat((1, 1, sample_num, 1))
            data[f'{mode}_out_pose_tp'] = torch.cat((root_rot, data[f'{mode}_out_body_pose_tp']), dim=-1)
        else:
            data[f'{mode}_out_pose_tp'] = x
            data[f'{mode}_out_body_pose_tp'] = x[..., 3:]

        if self.use_jpos:
            x_jpos = x_all[..., 69:138]
            if mode in {'recon', 'train'}:
                x_jpos = x_jpos.squeeze(2)
            data[f'{mode}_out_joint_pos_tp'] = x_jpos
            data[f'{mode}_out_joint_pos_frompose_tp'] = self.ctx['root_model'].get_joint_pos(data[f'{mode}_out_body_pose_tp'])
        if self.use_jvel:
            x_jvel = x_all[..., -69:]
            if mode in {'recon', 'train'}:
                x_jvel = x_jvel.squeeze(2)
            data[f'{mode}_out_joint_vel_tp'] = x_jvel


""" 
Main Model
"""

class MotionInfillerVAE(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.model_type = 'angle'
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
        self.past_nframe = specs['past_nframe']
        self.cur_nframe = specs['cur_nframe']
        self.fut_nframe = specs['fut_nframe']
        self.use_joints = specs.get('use_joints', False)
        self.pose_dropout = specs.get('pose_dropout', 0.0)
        pose_rep = specs.get('pose_rep', 'full')
        self.ctx = {
            'root_model': self,
            'nz': self.nz,
            'past_nframe': self.past_nframe,
            'cur_nframe': self.cur_nframe,
            'fut_nframe': self.fut_nframe,
            'pose_rep': pose_rep,
            'mlp_htype': specs['mlp_htype']
        }
        self.smpl = SMPL(SMPL_MODEL_DIR, pose_type='body26fk', create_transl=False)
        self.context_encoder = ContextEncoder(self.cfg, specs['context_encoder'], self.ctx)
        self.data_encoder = DataEncoder(self.cfg, specs['data_encoder'], self.ctx)
        self.data_decoder = DataDecoder(self.cfg, specs['data_decoder'], self.ctx)
        self.traj_predictor = None

    def forward(self, data):
        self.context_encoder(data)
        self.data_encoder(data)
        self.data_decoder(data, mode='train')
        return data

    def get_joint_pos(self, body_pose):
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
        # frame mask to be used by transformers
        data['invis_frame_mask'] = data['frame_mask'] == 1     # invisble frames are marked as False
        data['vis_frame_mask'] = ~data['invis_frame_mask']
        frame_mask = data['frame_mask'].transpose(0, 1).unsqueeze(-1)
        if 'frame_loss_mask' in data:
            data['frame_loss_mask_tp'] = data['frame_loss_mask'].transpose(0, 1)
        # pose
        if 'pose' in data:
            data['pose_tp'] = data['pose'].transpose(0, 1).contiguous()
            data['body_pose_tp'] = data['pose_tp'][..., 3:]
            if self.use_joints:
                data['joint_pos_tp'] = self.get_joint_pos(data['body_pose_tp'])
                data['joint_vel_tp'] = (data['joint_pos_tp'][1:] - data['joint_pos_tp'][:-1]) * 30  # 30 fps
                data['joint_vel_tp'] = torch.cat([data['joint_vel_tp'][[0]], data['joint_vel_tp']], dim=0)
            if self.data_decoder.rot_type == '6d':
                x = data['body_pose_tp']
                aa = x.view(x.shape[:-1] + (-1, 3))
                sixd = angle_axis_to_rot6d(aa)
                data['body_pose_6d_tp'] = sixd.view(x.shape[:-1] + (-1,))
                
        if 'pose_mask' in data:
            data['pose_mask_tp'] = data['pose_mask'].transpose(0, 1).contiguous()
        # in_pose
        if 'in_pose' not in data:
            if 'pose' in data:
                data['in_pose_tp'] = data['pose_tp'] * data['pose_mask_tp']
        else:
            data['in_pose_tp'] = data['in_pose'].transpose(0, 1).contiguous()
        # in_body_pose
        if 'in_body_pose' not in data:
            data['in_body_pose_tp'] = data['in_pose_tp'][..., 3:]
        else:
            data['in_body_pose_tp'] = data['in_body_pose'].transpose(0, 1).contiguous()
        # pose dropout
        if self.training and self.pose_dropout > 0:
            dropout_mask = torch.rand(data['in_body_pose_tp'].shape[:-1] + (23,), device=data['in_body_pose_tp'].device)
            dropout_mask = (dropout_mask > self.pose_dropout).float().repeat_interleave(3, dim=-1)
            data['in_body_pose_tp'] *= dropout_mask

        # in_joint_pos
        if self.use_joints:
            if 'joint_pos_tp' in data:
                data['in_joint_pos_tp'] = data['joint_pos_tp'].clone()
                data['in_joint_vel_tp'] = data['joint_vel_tp'].clone()
            else:
                data['in_joint_pos_tp'] = self.get_joint_pos(data['in_body_pose_tp'])
                data['in_joint_vel_tp'] = (data['in_joint_pos_tp'][1:] - data['in_joint_pos_tp'][:-1]) * 30  # 30 fps
                data['in_joint_vel_tp'] = torch.cat([data['in_joint_vel_tp'][[0]], data['in_joint_vel_tp']], dim=0)
            data['in_joint_pos_tp'] *= frame_mask
            data['in_joint_vel_tp'] *= frame_mask
        data['batch_size'] = data['in_body_pose_tp'].shape[1]
        data['seq_len'] = data['in_body_pose_tp'].shape[0]
        return data

    def inference_one_step(self, data, sample_num, recon):
        self.context_encoder(data)
        if recon:
            self.data_encoder(data)
            self.data_decoder(data, mode='recon')
            data['recon_out_pose'] = data['recon_out_pose_tp'].transpose(1, 0).contiguous()
            data['recon_out_body_pose'] = data['recon_out_pose'][..., 3:]
        else:
            self.data_decoder(data, mode='infer', sample_num=sample_num)
            data['infer_out_pose'] = data['infer_out_pose_tp'].permute(1, 2, 0, 3).contiguous()
            data['infer_out_body_pose'] = data['infer_out_pose'][..., 3:]
        return data

    def get_seg_data(self, data, seg_ind, sind, eind):
        data_i = dict()
        window_len = eind - sind
        data_i['batch_size'] = data['batch_size']
        data_i['seq_len'] = window_len
        if 'in_motion_latent' in data:
            data_i['in_motion_latent'] = data['in_motion_latent'][[seg_ind]]
        eind_b = min(eind, data['seq_len'])
        pad_len = eind - eind_b

        for key in data.keys():
            if 'tp' in key:
                data_i[key] = data[key][sind: eind_b].clone()
                if pad_len > 0:
                    pad = torch.zeros((pad_len,) + data[key].shape[1:], device=data[key].device, dtype=data[key].dtype)
                    data_i[key] = torch.cat([data_i[key], pad], dim=0)

        key = 'vis_frame_mask'
        data_i[key] = data[key][:, sind: eind_b].clone()
        if pad_len > 0:
            pad = torch.ones(data[key].shape[:-1] + (pad_len,), device=data[key].device, dtype=data[key].dtype)
            data_i[key] = torch.cat([data_i[key], pad], dim=1)

        return data_i

    def get_res_from_cur_data(self, data, cur_data, sind, eind, recon):
        mode = 'recon' if recon else 'infer'
        num_fr = min(eind - self.fut_nframe, data['seq_len']) - sind

        if mode == 'recon':
            for key in ['pose', 'body_pose']:
                if f'in_{key}_tp' in data:
                    # assert torch.all(data[f'in_{key}_tp'][sind: sind + self.past_nframe] == cur_data[f'recon_out_{key}_tp'][:self.past_nframe])
                    data[f'in_{key}_tp'][sind: sind + num_fr] = cur_data[f'recon_out_{key}_tp'][:num_fr]

                    if f'recon_out_{key}' not in data:
                        data[f'recon_out_{key}'] = cur_data[f'recon_out_{key}'][:, :num_fr]
                    else:
                        data[f'recon_out_{key}'] = torch.cat([data[f'recon_out_{key}'], cur_data[f'recon_out_{key}'][:, self.past_nframe: num_fr]], dim=1)
        else:
            for key in ['pose', 'body_pose']:
                if f'in_{key}_tp' in data:
                    # assert torch.all(data[f'in_{key}_tp'][sind: sind + self.past_nframe] == cur_data[f'infer_out_{key}_tp'][:self.past_nframe, :, 0])
                    data[f'in_{key}_tp'][sind: sind + num_fr] = cur_data[f'infer_out_{key}_tp'][:num_fr, :, 0]
                    if f'infer_out_{key}' not in data:
                        data[f'infer_out_{key}'] = cur_data[f'infer_out_{key}'][:, [0], :num_fr]
                    else:
                        data[f'infer_out_{key}'] = torch.cat([data[f'infer_out_{key}'], cur_data[f'infer_out_{key}'][:, [0], self.past_nframe: num_fr]], dim=2)

    def get_latent(self, seq_len):
        num_latent = int(np.ceil((seq_len - self.past_nframe) / self.cur_nframe))
        latent = torch.randn((num_latent, self.nz))
        return latent

    def inference_multi_step(self, batch, sample_num, recon):
        data = self.init_batch_data(batch)
        past_nframe = self.past_nframe
        cur_nframe = self.cur_nframe
        fut_nframe = self.fut_nframe
        window_len = past_nframe + cur_nframe + fut_nframe
        total_len = data['in_body_pose_tp'].shape[0]
        for i in range(int(np.ceil((total_len - past_nframe) / cur_nframe))):
            sind = i * cur_nframe
            eind = i * cur_nframe + window_len
            data_i = self.get_seg_data(data, i, sind, eind)
            data_i['vis_frame_mask'][:, :self.past_nframe] = False
            self.inference_one_step(data_i, sample_num, recon)
            self.get_res_from_cur_data(data, data_i, sind, eind, recon)
        return data

    def merge_infer_samples(self, data_list):
        if len(data_list) == 1:
            return data_list[0]
        data = data_list[0]
        data['infer_out_body_pose'] = torch.cat([x['infer_out_body_pose'] for x in data_list], dim=1)
        if 'infer_out_pose' in data:
            data['infer_out_pose'] = torch.cat([x['infer_out_pose'] for x in data_list], dim=1)
        return data

    def inference(self, batch, sample_num=5, recon=False, multi_step=False):
        if multi_step:
            data = []
            for _ in range(sample_num):
                data.append(self.inference_multi_step(batch, sample_num=1, recon=False))
            data = self.merge_infer_samples(data)
            if recon:
                data_recon = self.inference_multi_step(batch, sample_num, recon=True)
                data['recon_out_pose'] = data_recon['recon_out_pose']
                data['recon_out_body_pose'] = data_recon['recon_out_body_pose']
        else:
            data = self.init_batch_data(batch)
            self.context_encoder(data)
            self.data_decoder(data, mode='infer', sample_num=sample_num)
            data['infer_out_pose'] = data['infer_out_pose_tp'].permute(1, 2, 0, 3).contiguous()
            data['infer_out_body_pose'] = data['infer_out_pose'][..., 3:]
            if recon:
                self.data_encoder(data)
                self.data_decoder(data, mode='recon')
                data['recon_out_pose'] = data['recon_out_pose_tp'].transpose(1, 0).contiguous()
                data['recon_out_body_pose'] = data['recon_out_pose'][..., 3:]
            data['pose'] = data['pose'][:, :-self.fut_nframe]
            data['trans'] = data['trans'][:, :-self.fut_nframe]
            data['shape'] = data['shape'][:, :-self.fut_nframe]
        return data

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
