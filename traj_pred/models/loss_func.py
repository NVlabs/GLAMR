from lib.utils.torch_transform import quat_angle_diff, get_heading, rot6d_to_quat, vec_to_heading



def compute_trans_mse(data, specs):
    mode = specs.get('mode', 'train')
    use_frame_loss_mask = specs.get('use_frame_loss_mask', False)
    diff = data[f'{mode}_out_trans_tp'] - data[f'trans_tp']
    if use_frame_loss_mask:
        diff *= data['frame_loss_mask'].transpose(0, 1)
    mse = diff.pow(2).sum(-1).mean()
    return mse


def compute_orient_angle_loss(data, specs):
    mode = specs.get('mode', 'train')
    use_frame_loss_mask = specs.get('use_frame_loss_mask', False)
    angle = quat_angle_diff(data[f'{mode}_out_orient_q_tp'], data[f'orient_q_tp'])
    if use_frame_loss_mask:
        angle *= data['frame_loss_mask'].transpose(0, 1).squeeze(-1)
    angle_loss = angle.pow(2).mean()
    return angle_loss


def compute_orient_6d_loss(data, specs):
    mode = specs.get('mode', 'train')
    use_frame_loss_mask = specs.get('use_frame_loss_mask', False)
    diff = data[f'{mode}_out_orient_6d_tp'] - data[f'orient_6d_tp']
    if use_frame_loss_mask:
        diff *= data['frame_loss_mask'].transpose(0, 1)
    mse = diff.pow(2).sum(-1).mean()
    return mse


def compute_vae_z_kld(data, specs):
    clamp_before_mean = specs.get('clamp_before_mean', True)
    kld = data['q_z_dist'].kl(data['p_z_dist'])
    kld = kld.sum(-1)
    if clamp_before_mean:
        kld = kld.clamp_min_(specs['min_clip']).mean()
    else:
        kld = kld.mean().clamp_min_(specs['min_clip'])
    return kld


def compute_local_orient_heading(data, specs):
    local_traj = data[f'train_out_local_traj_tp']
    local_orient = local_traj[..., 3:-2]
    if local_orient.shape[-1] == 6:
        local_orient = rot6d_to_quat(local_orient)
    heading = get_heading(local_orient)
    mse = heading.pow(2).mean()
    return mse


def compute_dheading(data, specs):
    local_traj = data[f'train_out_local_traj_tp']
    local_heading_vec = local_traj[..., -2:]
    heading = vec_to_heading(local_heading_vec)
    mse = heading.pow(2).mean()
    return mse


loss_func_dict = {
    'trans_mse': compute_trans_mse,
    'orient_angle': compute_orient_angle_loss,
    'vae_z_kld': compute_vae_z_kld,
    'orient_6d': compute_orient_6d_loss,
    'local_orient_heading': compute_local_orient_heading,
    'dheading': compute_dheading
}