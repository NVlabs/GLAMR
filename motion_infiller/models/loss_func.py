from lib.utils.torch_transform import angle_axis_to_quaternion, quat_angle_diff, vec_to_heading
from lib.utils.dist import Normal


def compute_nll(data, specs):
    nll = data['nll'].mean()
    return nll
    

def compute_mse(data, specs):
    key = 'body_pose' if specs.get('body_only', False) else 'pose'
    vis_only = specs.get('vis_only', False)
    num_fr = data[f'train_out_{key}_tp'].shape[0]
    diff = data[f'train_out_{key}_tp'] - data[f'{key}_tp'][:num_fr]
    # diff *= data['frame_loss_mask_tp'][:num_fr]
    dist = diff.pow(2).sum(-1)
    if vis_only:
        dist *= 1 - data['frame_mask'].transpose(0, 1)
    mse = dist.mean()
    return mse


def compute_rot6d_mse(data, specs):
    diff = data[f'train_out_body_pose_6d_tp'] - data[f'body_pose_6d_tp']
    # diff *= data['frame_loss_mask_tp']
    mse = diff.pow(2).sum(-1).mean()
    return mse


def compute_jpos_gt_loss(data, specs):
    num_fr = data[f'train_out_joint_pos_tp'].shape[0]
    diff = data[f'train_out_joint_pos_tp'] - data[f'joint_pos_tp'][:num_fr]
    # diff *= data['frame_loss_mask_tp']
    mse = diff.pow(2).sum(-1).mean()
    return mse


def compute_jvel_gt_loss(data, specs):
    num_fr = data[f'train_out_joint_vel_tp'].shape[0]
    diff = data[f'train_out_joint_vel_tp'] - data[f'joint_vel_tp'][:num_fr]
    # diff *= data['frame_loss_mask_tp']
    mse = diff.pow(2).sum(-1).mean()
    return mse


def compute_jpos_frompose_gt_loss(data, specs):
    diff = data[f'train_out_joint_pos_frompose_tp'] - data[f'joint_pos_tp']
    # diff *= data['frame_loss_mask_tp']
    mse = diff.pow(2).sum(-1).mean()
    return mse


def compute_jpos_consist_loss(data, specs):
    diff = data[f'train_out_joint_pos_frompose_tp'] - data[f'train_out_joint_pos_tp']
    # diff *= data['frame_loss_mask_tp']
    mse = diff.pow(2).sum(-1).mean()
    return mse


def compute_vae_z_kld(data, specs):
    clamp_before_mean = specs.get('clamp_before_mean', True)
    vis_only = specs.get('vis_only', False)
    kld = data['q_z_dist'].kl(data['p_z_dist'])
    kld = kld.sum(-1)
    if vis_only:
        kld *= 1 - data['frame_mask'].transpose(0, 1)
    if clamp_before_mean:
        kld = kld.clamp_min_(specs['min_clip']).mean()
    else:
        kld = kld.mean().clamp_min_(specs['min_clip'])
    return kld


def compute_vae_z_kld_bidir(data, specs):
    kld_forward = data['q_z_dist_forward'].kl(data['p_z_dist_forward'])
    kld_backward = data['q_z_dist_backward'].kl(data['p_z_dist_backward'])
    kld = (kld_forward.sum(-1) + kld_backward.sum(-1)) * 0.5
    kld = kld.clamp_min_(specs['min_clip']).mean()
    return kld


def compute_vae_z_prior_smoothness(data, specs):
    dist1 = Normal(mu=data['p_z_dist'].mu[1:], logvar=data['p_z_dist'].logvar[1:])
    dist2 = Normal(mu=data['p_z_dist'].mu[:-1], logvar=data['p_z_dist'].logvar[:-1])
    kld = dist1.kl(dist2)
    kld = kld.sum(-1).mean()
    return kld


def compute_vae_z_posterior_smoothness(data, specs):
    dist1 = Normal(mu=data['q_z_dist'].mu[1:], logvar=data['q_z_dist'].logvar[1:])
    dist2 = Normal(mu=data['q_z_dist'].mu[:-1], logvar=data['q_z_dist'].logvar[:-1])
    kld = dist1.kl(dist2)
    kld = kld.sum(-1).mean()
    return kld


def compute_axis_angle_quat_smoothness(data, specs):
    key = 'body_pose' if specs.get('body_only', False) else 'pose'
    diff = data[f'train_out_{key}_tp'] - data[f'{key}_tp']
    # diff *= data['frame_loss_mask'].transpose(0, 1)
    mse = diff.pow(2).sum(-1).mean()
    return mse


def compute_trans_mse(data, specs):
    mode = specs.get('mode', 'train')
    use_frame_loss_mask = specs.get('use_frame_loss_mask', False)
    num_fr = data[f'{mode}_out_trans_tp'].shape[0]
    diff = data[f'{mode}_out_trans_tp'] - data[f'trans_tp'][:num_fr]
    if use_frame_loss_mask:
        diff *= data['frame_loss_mask'].transpose(0, 1)
    mse = diff.pow(2).sum(-1).mean()
    return mse


def compute_orient_angle_loss(data, specs):
    mode = specs.get('mode', 'train')
    use_frame_loss_mask = specs.get('use_frame_loss_mask', False)
    num_fr = data[f'{mode}_out_orient_q_tp'].shape[0]
    angle = quat_angle_diff(data[f'{mode}_out_orient_q_tp'], data[f'orient_q_tp'][:num_fr])
    if use_frame_loss_mask:
        angle *= data['frame_loss_mask'].transpose(0, 1).squeeze(-1)
    angle_loss = angle.pow(2).mean()
    return angle_loss


def compute_orient_6d_loss(data, specs):
    mode = specs.get('mode', 'train')
    use_frame_loss_mask = specs.get('use_frame_loss_mask', False)
    num_fr = data[f'{mode}_out_orient_6d_tp'].shape[0]
    diff = data[f'{mode}_out_orient_6d_tp'] - data[f'orient_6d_tp'][:num_fr]
    if use_frame_loss_mask:
        diff *= data['frame_loss_mask'].transpose(0, 1)
    mse = diff.pow(2).sum(-1).mean()
    return mse


def compute_local_orient_smoothness(data, specs):
    local_traj = data[f'train_out_local_traj_tp']
    local_orient = local_traj[..., 3:-2]
    diff = local_orient[1:] - local_orient[:-1]
    mse = diff.pow(2).sum(-1).mean()
    return mse


def compute_dheading_smoothness(data, specs):
    local_traj = data[f'train_out_local_traj_tp']
    d_heading_vec = local_traj[1:, :, 9:]
    d_heading = vec_to_heading(d_heading_vec)
    mse = d_heading.pow(2).mean()
    return mse


loss_func_dict = {
    'nll': compute_nll,
    'mse': compute_mse,
    'rot6d_mse': compute_rot6d_mse,
    'jpos_gt': compute_jpos_gt_loss,
    'jvel_gt': compute_jvel_gt_loss,
    'jpos_frompose': compute_jpos_frompose_gt_loss,
    'jpos_consist': compute_jpos_consist_loss,
    'vae_z_kld': compute_vae_z_kld,
    'vae_z_kld_bidir': compute_vae_z_kld_bidir,
    'vae_p_z_sm': compute_vae_z_prior_smoothness,
    'vae_q_z_sm': compute_vae_z_posterior_smoothness,
    'trans_mse': compute_trans_mse,
    'orient_angle': compute_orient_angle_loss,
    'orient_6d': compute_orient_6d_loss,
    'local_orient_sm': compute_local_orient_smoothness,
    'dheading_sm': compute_dheading_smoothness
}