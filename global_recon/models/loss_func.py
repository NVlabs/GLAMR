import torch
import numpy as np
from lib.utils.torch_transform import heading_to_vec, quat_angle_diff, rotmat_to_rot6d, angle_axis_to_rot6d, angle_axis_to_quaternion, inverse_transform, rotation_matrix_to_quaternion


def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def keypoint_2d_loss(data, specs):
    loss_all = 0
    num_pose = 0
    min_conf = specs.get('min_conf', 0.05)
    first_frame_only = specs.get('first_frame_only', False)
    first_frame_weight = specs.get('first_frame_weight', 1.0)
    for pose_dict in data['person_data'].values():
        vis_frames = pose_dict['vis_frames']
        diff = pose_dict['kp_2d_pred'][vis_frames] - pose_dict['kp_2d_aligned'][vis_frames]
        score = pose_dict['kp_2d_score'][vis_frames].clone()
        score[score < min_conf] = 0
        loss = gmof(diff, sigma=100)
        if first_frame_only:
            loss = loss[[0]]
            num_pose += vis_frames.sum()
        else:
            num_pose += vis_frames.sum()
        loss[:10] *= first_frame_weight
        loss = (loss.sum(-1) * (score ** 2)).sum()
        loss_all += loss
    loss_all /= num_pose
    return loss_all


def keypoint_2d_dist(data, specs):
    dist_all = []
    min_conf = specs.get('min_conf', 0.05)
    first_frame_only = specs.get('first_frame_only', False)
    for pose_dict in data['person_data'].values():
        kp_2d_score = pose_dict['kp_2d_score']
        kp_2d_pred = pose_dict['kp_2d_pred']
        kp_2d_aligned = pose_dict['kp_2d_aligned']
        if first_frame_only:
            kp_2d_score = kp_2d_score[[0]]
            kp_2d_pred = kp_2d_pred[[0]]
            kp_2d_aligned = kp_2d_aligned[[0]]
        sind = (kp_2d_score > min_conf).view(-1)
        diff = kp_2d_pred - kp_2d_aligned
        dist = diff.pow(2).sum(-1).sqrt().view(-1)[sind]
        dist_all.append(dist)
    dist_all = torch.cat(dist_all)
    dist_all = dist_all.mean()
    return dist_all


def cam_rot_smoothness_loss(data, specs):
    cam_rot1 = data['cam_rot_6d'][1:]
    cam_rot2 = data['cam_rot_6d'][:-1]
    vel = (cam_rot2 - cam_rot1) * 30    # 30 fps
    loss = vel.pow(2).sum(-1).mean()
    return loss


def cam_trans_smoothness_loss(data, specs):
    cam_trans1 = data['cam_trans'][1:]
    cam_trans2 = data['cam_trans'][:-1]
    vel = (cam_trans2 - cam_trans1) * 30    # 30 fps
    loss = vel.pow(2).sum(-1).mean()
    return loss


def cam_inv_rot_smoothness_loss(data, specs):
    cam_rot1 = data['cam_pose_inv'][1:, :3, :2]
    cam_rot2 = data['cam_pose_inv'][:-1, :3, :2]
    vel = (cam_rot2 - cam_rot1) * 30    # 30 fps
    loss = vel.pow(2).sum(-1).sum(-1).mean()
    return loss


def cam_origin_smoothness_loss(data, specs):
    cam_to_world1 = data['cam_pose_inv'][1:]
    cam_to_world2 = data['cam_pose_inv'][:-1]
    cam_orig1 = cam_to_world1[:, :3, 3]
    cam_orig2 = cam_to_world2[:, :3, 3]
    vel = (cam_orig1 - cam_orig2) * 30    # 30 fps
    loss = vel.pow(2).sum(-1).mean()
    return loss


def cam_depth_smoothness_loss(data, specs):
    cam_to_world1 = data['cam_pose_inv'][1:]
    cam_to_world2 = data['cam_pose_inv'][:-1]
    cam_orig1 = cam_to_world1[:, :3, 3]
    cam_orig2 = cam_to_world2[:, :3, 3]
    cam_z = cam_to_world1[:, :3, 2]
    cam_delta_z = ((cam_orig2 - cam_orig1) * cam_z).sum(-1)
    vel = cam_delta_z * 30    # 30 fps
    loss = vel.pow(2).sum(-1).mean()
    return loss


def cam_up_reg(data, specs):
    first_frame_weight = specs.get('first_frame_weight', 1.0)
    first_frame_only = specs.get('first_frame_only', False)
    cam_up_dot_z = data['cam_pose_inv'][:, 2, 1].clone()
    cam_up_dot_z[:10] *= first_frame_weight
    if first_frame_only:
        cam_up_dot_z = cam_up_dot_z[[0]]
    loss = cam_up_dot_z.mean()
    return loss


def traj_rot_smoothness_loss(data, specs):
    loss_all = 0
    num_pose = 0
    rot_type = specs.get('rot_type', '6d')
    for pose_dict in data['person_data'].values():
        num_pose += pose_dict['smpl_orient_world'].shape[0] - 1
        if rot_type == '6d':
            orient_6d = angle_axis_to_rot6d(pose_dict['smpl_orient_world'])
            diff = orient_6d[1:] - orient_6d[:-1]
        elif rot_type == 'quat':
            quat = angle_axis_to_quaternion(pose_dict['smpl_orient_world'])
            diff = quat_angle_diff(quat[1:], quat[:-1])
        vel = diff * 30
        loss_all += vel.pow(2).sum()
    loss_all /= num_pose
    return loss_all


def traj_trans_smoothness_loss(data, specs):
    loss_all = 0
    num_pose = 0
    for pose_dict in data['person_data'].values():
        num_pose += pose_dict['root_trans_world'].shape[0] - 1
        diff = pose_dict['root_trans_world'][1:] - pose_dict['root_trans_world'][:-1]
        vel = diff * 30
        loss_all += vel.pow(2).sum()
    loss_all /= num_pose
    return loss_all


def cam_traj_rot_loss(data, specs):
    loss_all = 0
    num_pose = 0
    rot_type = specs.get('rot_type', '6d')
    first_frame_weight = specs.get('first_frame_weight', 1.0)
    first_frame_only = specs.get('first_frame_only', False)
    for pose_dict in data['person_data'].values():
        vis_frames = pose_dict['vis_frames']
        if rot_type == '6d':
            rot1 = angle_axis_to_rot6d(pose_dict['smpl_orient_cam_in_world'][vis_frames])
            rot2 = angle_axis_to_rot6d(pose_dict['smpl_orient_cam'][vis_frames])
            diff = rot2 - rot1 
        elif rot_type == 'quat':
            rot1 = angle_axis_to_quaternion(pose_dict['smpl_orient_cam_in_world'][vis_frames])
            rot2 = angle_axis_to_quaternion(pose_dict['smpl_orient_cam'][vis_frames])
            diff = quat_angle_diff(rot2, rot1)

        if first_frame_only:
            diff = diff[[0]]
            num_pose += 1
        else:
            diff[0] *= first_frame_weight
            num_pose += vis_frames.sum()
        loss_all += diff.pow(2).sum()
    loss_all /= num_pose
    return loss_all


def cam_traj_trans_loss(data, specs):
    loss_all = 0
    num_pose = 0
    first_frame_weight = specs.get('first_frame_weight', 1.0)
    for pose_dict in data['person_data'].values():
        vis_frames = pose_dict['vis_frames']
        num_pose += vis_frames.sum()
        diff = pose_dict['root_trans_cam_in_world'][vis_frames] - pose_dict['root_trans_cam'][vis_frames]
        diff[0] *= first_frame_weight
        loss_all += diff.pow(2).sum()
    loss_all /= num_pose
    return loss_all


def reg_loss(data, key):
    loss_all = 0
    num_pose = 0
    for pose_dict in data['person_data'].values():
        num_pose += pose_dict[key].shape[0]
        loss_all += (pose_dict[key] * 30).pow(2).sum()
    loss_all /= num_pose
    return loss_all


def reg_loss_global(data, key):
    loss_all = (data[key] * 30).pow(2).sum() / data[key].shape[0]
    return loss_all


def traj_rot_res_loss(data, specs):
    return reg_loss(data, 'smpl_orient_world_res')


def traj_trans_res_loss(data, specs):
    return reg_loss(data, 'root_trans_world_res')


def local_traj_dxy_reg(data, specs):
    return reg_loss(data, 'traj_local_dxy')


def local_traj_dheading_reg(data, specs):
    return reg_loss(data, 'traj_local_dheading')


def local_traj_dheading_reg_new(data, specs):
    loss_all = 0
    num_pose = 0
    key = 'traj_local_dheading'
    for pose_dict in data['person_data'].values():
        num_pose += pose_dict[key].shape[0]
        vec = heading_to_vec(pose_dict[key])
        diff = vec - torch.tensor([1.0, 0.0]).type_as(vec)
        loss_all += (diff * 30).pow(2).sum()
    loss_all /= num_pose
    return loss_all

def local_traj_rot_reg(data, specs):
    return reg_loss(data, 'traj_local_rot')


def local_traj_z_reg(data, specs):
    return reg_loss(data, 'traj_local_z')


def cam_inv_trans_residual_reg(data, specs):
    return reg_loss_global(data, 'cam_inv_trans_residual')


def person2cam_res_trans_reg(data, specs):
    return reg_loss_global(data, 'person2cam_res_trans')


def rel_transform_loss(data, specs):
    loss_all = 0
    num_pose = 0
    person_data = data['person_data']
    trans_weight = specs.get('trans_weight', 1.0)
    first_frame_weight = specs.get('first_frame_weight', 10)
    first_frame_trans_only = specs.get('first_frame_trans_only', False)
    for (i, j), rel_transform_cam in data['rel_transform_cam'].items():
        num_pose += rel_transform_cam.shape[0]
        vis_frames = person_data[i]['vis_frames'] & person_data[j]['vis_frames']
        if sum(vis_frames) == 0:
            continue
        rel_transform_world = torch.matmul(inverse_transform(person_data[i]['person_transform_world'][vis_frames]), person_data[j]['person_transform_world'][vis_frames])
        rel_transform_cam = rel_transform_cam[vis_frames]
        diff_rot = rel_transform_cam[..., :3, :2] - rel_transform_world[..., :3, :2]
        diff_trans = rel_transform_cam[..., :3, 3] - rel_transform_world[..., :3, 3]
        diff_rot[0] *= first_frame_weight
        diff_trans[0] *= first_frame_weight
        if first_frame_trans_only:
            diff_trans[1:] = 0.0
        loss_all += diff_rot.pow(2).sum() + diff_trans.pow(2).sum() * trans_weight
    if num_pose > 0:
        loss_all /= num_pose
    return loss_all


def penetration_loss(data, specs):
    loss_all = 0
    sdf_loss = data['sdf_loss']
    person_data = data['person_data']
    max_fr = person_data[0]['max_len']
    for fr in range(max_fr):
        verts = []
        for idx in person_data.keys():
            if person_data[idx]['vis_frames'][fr]:
                verts.append(person_data[idx]['smpl_verts'][fr])
        if len(verts) < 2:
            continue
        verts = torch.stack(verts)
        loss = sdf_loss(verts, translation=torch.zeros((verts.shape[0], 3)).type_as(verts))
        loss_all += loss
    loss_all /= max_fr
    return loss_all


def motion_latent_reg_loss(data, specs):
    loss_all = 0
    num_latent = 0
    for pose_dict in data['person_data'].values():
        num_latent += pose_dict['motion_latent'].shape[0]
        loss_all += pose_dict['motion_latent'].pow(2).sum()
    loss_all /= num_latent
    return loss_all


def traj_latent_reg_loss(data, specs):
    loss_all = 0
    num_latent = 0
    for pose_dict in data['person_data'].values():
        num_latent += pose_dict['traj_latent'].shape[0]
        loss_all += pose_dict['traj_latent'].pow(2).sum()
    loss_all /= num_latent
    return loss_all



loss_func_dict = {
    'kp_2d': keypoint_2d_loss,
    'kp_2d_dist': keypoint_2d_dist,
    'cam_rot_smoothness': cam_rot_smoothness_loss,
    'cam_trans_smoothness': cam_trans_smoothness_loss,
    'cam_inv_rot_smoothness': cam_inv_rot_smoothness_loss,
    'cam_origin_smoothness': cam_origin_smoothness_loss,
    'cam_depth_smoothness': cam_depth_smoothness_loss,
    'traj_rot_smoothness': traj_rot_smoothness_loss,
    'traj_trans_smoothness': traj_trans_smoothness_loss,
    'cam_up_reg': cam_up_reg,
    'cam_traj_rot': cam_traj_rot_loss,
    'cam_traj_trans': cam_traj_trans_loss,
    'traj_rot_res': traj_rot_res_loss,
    'traj_trans_res': traj_trans_res_loss,
    'local_traj_dxy_reg': local_traj_dxy_reg,
    'local_traj_dheading_reg': local_traj_dheading_reg,
    'local_traj_dheading_reg_new': local_traj_dheading_reg_new,
    'local_traj_rot_reg': local_traj_rot_reg,
    'local_traj_z_reg': local_traj_z_reg,
    'cam_inv_trans_residual_reg': cam_inv_trans_residual_reg,
    'person2cam_res_trans_reg': person2cam_res_trans_reg,
    'rel_transform': rel_transform_loss,
    'motion_latent_reg': motion_latent_reg_loss,
    'traj_latent_reg': traj_latent_reg_loss,
    'penetration': penetration_loss
}