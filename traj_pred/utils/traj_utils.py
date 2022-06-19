import torch
import numpy as np
from lib.utils.torch_transform import get_heading_q, quat_apply, heading_to_quat, heading_to_vec, normalize, vec_to_heading, ypr_euler_from_quat, quat_from_ypr_euler, quat_mul, quat_conjugate, get_heading, deheading_quat, quat_to_rot6d, rot6d_to_quat
from scipy.interpolate import interp1d


def rot_2d(xy, theta):
    rot_x = xy[..., 0] * torch.cos(theta) - xy[..., 1] * torch.sin(theta)
    rot_y = xy[..., 0] * torch.sin(theta) + xy[..., 1] * torch.cos(theta)
    rot_xy = torch.stack([rot_x, rot_y], dim=-1)
    return rot_xy


def traj_global2local(trans, orient_q, base_orient=[0.5, 0.5, 0.5, 0.5]):
    base_orient = torch.tensor(base_orient, device=orient_q.device)
    xy, z = trans[..., :2], trans[..., 2]
    orient_q = quat_mul(orient_q, quat_conjugate(base_orient).expand_as(orient_q))
    eulers = ypr_euler_from_quat(orient_q)
    roll, pitch, yaw = eulers[..., 0], eulers[..., 1], eulers[..., 2]
    d_xy = xy[1:] - xy[:-1]
    d_yaw = yaw[1:] - yaw[:-1]
    d_xy_yawcoord = rot_2d(d_xy, -yaw[:-1])
    d_xy_yawcoord = torch.cat([xy[[0]], d_xy_yawcoord])     # first element is global trans xy
    d_yaw = torch.cat([yaw[[0]], d_yaw])    # first element is global yaw
    local_traj = torch.stack([d_xy_yawcoord[..., 0], d_xy_yawcoord[..., 1], z, roll, pitch, d_yaw], dim=-1)
    return local_traj


def traj_local2global(local_traj, base_orient=[0.5, 0.5, 0.5, 0.5]):
    base_orient = torch.tensor(base_orient, device=local_traj.device)
    d_xy_yawcoord, z = local_traj[..., :2], local_traj[..., 2]
    roll, pitch, d_yaw = local_traj[..., 3], local_traj[..., 4], local_traj[..., 5]
    yaw = torch.cumsum(d_yaw, dim=0)
    d_xy = d_xy_yawcoord.clone()
    d_xy[1:] = rot_2d(d_xy_yawcoord[1:], yaw[:-1])
    xy = torch.cumsum(d_xy, dim=0)
    trans = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
    eulers = torch.stack([roll, pitch, yaw], dim=-1)
    orient_q = quat_from_ypr_euler(eulers)
    orient_q = quat_mul(orient_q, base_orient.expand_as(orient_q))
    return trans, orient_q


def traj_global2local_heading(trans, orient_q, base_orient=[0.5, 0.5, 0.5, 0.5], local_orient_type='6d'):
    base_orient = torch.tensor(base_orient, device=orient_q.device)
    xy, z = trans[..., :2], trans[..., 2]
    orient_q = quat_mul(orient_q, quat_conjugate(base_orient).expand_as(orient_q))
    heading = get_heading(orient_q)
    heading_q = get_heading_q(orient_q)
    local_q = deheading_quat(orient_q, heading_q)
    if local_orient_type == '6d':
        local_orient = quat_to_rot6d(local_q)
    else:
        local_orient = local_q[..., :3]
    d_xy = xy[1:] - xy[:-1]
    d_heading = heading[1:] - heading[:-1]
    d_heading = torch.cat([heading[[0]], d_heading])    # first element is global heading
    d_heading_vec = heading_to_vec(d_heading)
    d_xy_yawcoord = rot_2d(d_xy, -heading[:-1])
    d_xy_yawcoord = torch.cat([xy[[0]], d_xy_yawcoord])     # first element is global trans xy
    local_traj = torch.cat([d_xy_yawcoord[..., :2], z.unsqueeze(-1), local_orient, d_heading_vec], dim=-1)  # dim: 3 + 6 + 2 = 11
    return local_traj


def traj_local2global_heading(local_traj, base_orient=[0.5, 0.5, 0.5, 0.5], deheading_local=False, local_orient_type='6d', local_heading=True):
    base_orient = torch.tensor(base_orient, device=local_traj.device)
    d_xy_yawcoord, z = local_traj[..., :2], local_traj[..., 2]
    local_orient, d_heading_vec = local_traj[..., 3:-2], local_traj[..., -2:]
    d_heading = vec_to_heading(d_heading_vec)
    if local_heading:
        heading = torch.cumsum(d_heading, dim=0)
    else:
        heading = d_heading
    heading_q = heading_to_quat(heading)
    d_xy = d_xy_yawcoord.clone()
    d_xy[1:] = rot_2d(d_xy_yawcoord[1:], heading[:-1])
    xy = torch.cumsum(d_xy, dim=0)
    trans = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
    if local_orient_type == '6d':
        local_q = rot6d_to_quat(local_orient)
        if deheading_local:
            local_q = deheading_quat(local_q)
    else:
        local_q = torch.cat([local_orient, torch.zeros_like(local_orient[..., [0]])], dim=-1)
        local_q = normalize(local_q)
    orient_q = quat_mul(heading_q, local_q)
    orient_q = quat_mul(orient_q, base_orient.expand_as(orient_q))
    return trans, orient_q


def get_init_heading_q(orient, base_orient=[0.5, 0.5, 0.5, 0.5]):
    orient_nobase = quat_mul(orient[0], quat_conjugate(torch.tensor(base_orient, device=orient.device)).expand_as(orient[0]))
    heading_q = get_heading_q(orient_nobase)
    return heading_q


def convert_traj_world2heading(orient, trans, base_orient=[0.5, 0.5, 0.5, 0.5], apply_base_orient_after=False):
    orient_nobase = quat_mul(orient, quat_conjugate(torch.tensor(base_orient, device=orient.device)).expand_as(orient))
    heading_q = get_heading_q(orient_nobase[0])
    inv_heading_q = quat_conjugate(heading_q).expand_as(orient_nobase)
    orient_heading = quat_mul(inv_heading_q, orient_nobase)
    trans_local = trans.clone()
    trans_local[..., :2] -= trans[0, ..., :2]
    trans_heading = quat_apply(inv_heading_q, trans_local)
    if apply_base_orient_after:
        orient_heading = quat_mul(orient_heading, torch.tensor(base_orient, device=orient_heading.device).expand_as(orient_heading))
    return orient_heading, trans_heading


def convert_traj_heading2world(orient, trans, init_heading, init_trans, base_orient=[0.5, 0.5, 0.5, 0.5]):
    init_heading = init_heading.expand_as(orient)
    trans_local = quat_apply(init_heading, trans)
    trans_world = trans_local.clone()
    trans_world[..., :2] += init_trans[..., :2]
    orient_nobase = quat_mul(init_heading, orient)
    orient_world = quat_mul(orient_nobase, torch.tensor(base_orient, device=orient.device).expand_as(orient))
    return orient_world, trans_world


def interp_orient_q_sep_heading(orient_q_vis, vis_frames, base_orient=[0.5, 0.5, 0.5, 0.5]):
    device = orient_q_vis.device
    base_orient = torch.tensor(base_orient, device=device)
    orient_q_vis_rb = quat_mul(orient_q_vis, quat_conjugate(base_orient).expand_as(orient_q_vis))
    heading_q = get_heading_q(orient_q_vis_rb)
    heading_vec = heading_to_vec(get_heading(orient_q_vis_rb))
    local_orient = quat_to_rot6d(deheading_quat(orient_q_vis_rb, heading_q))
    max_len = vis_frames.shape[0]
    vis_ind = torch.where(vis_frames)[0].cpu().numpy()
    # heading_vec
    f = interp1d(vis_ind, heading_vec.cpu().numpy(), axis=0, assume_sorted=True, fill_value="extrapolate")
    new_val = f(np.arange(max_len, dtype=np.float32))
    heading_vec_interp = torch.tensor(new_val, device=device, dtype=torch.float32)
    # local_orient
    f = interp1d(vis_ind, local_orient.cpu().numpy(), axis=0, assume_sorted=True, fill_value="extrapolate")
    new_val = f(np.arange(max_len, dtype=np.float32))
    local_orient_interp = torch.tensor(new_val, device=device, dtype=torch.float32)
    # final
    heading_q_interp = heading_to_quat(vec_to_heading(heading_vec_interp))
    local_q_interp = rot6d_to_quat(local_orient_interp)
    orient_q_interp = quat_mul(heading_q_interp, local_q_interp)
    orient_q_interp = quat_mul(orient_q_interp, base_orient.expand_as(orient_q_interp))
    return orient_q_interp