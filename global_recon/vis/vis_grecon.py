import os, sys
sys.path.append(os.path.join(os.getcwd()))
import os.path as osp
import pyvista
import time
import torch
import numpy as np
import glob
import vtk
from collections import defaultdict
from lib.utils.visualizer3d import Visualizer3D
from lib.models.smpl import SMPL, SMPL_MODEL_DIR
from motion_infiller.vis.vis_smpl import SMPLActor, SkeletonActor
from traj_pred.utils.traj_utils import convert_traj_world2heading
from lib.utils.vis import hstack_videos, make_checker_board_texture, vtk_matrix_to_nparray
from lib.utils.torch_transform import angle_axis_to_quaternion, quaternion_to_angle_axis, quat_apply


class GReconVisualizer(Visualizer3D):

    def __init__(self, data, coord, device=torch.device('cpu'), use_y_up_coord=False, use_est_traj=False, background_img_dir=None,
                 show_gt_pose=False, show_est_pose=True, show_smpl=True, show_skeleton=False, show_camera=True, align_pose=False, view_dist=13.0, render_cam_pos=None, render_cam_focus=None, **kwargs):
        super().__init__(use_floor=False, **kwargs)
        self.device = device
        self.use_y_up_coord = use_y_up_coord
        self.use_est_traj = use_est_traj
        self.show_gt_pose = show_gt_pose
        self.show_est_pose = show_est_pose
        self.show_smpl = show_smpl
        self.show_skeleton = show_skeleton
        self.show_camera = show_camera
        self.align_pose = align_pose
        self.view_dist = view_dist
        self.render_cam_pos = render_cam_pos
        self.render_cam_focus = render_cam_focus
        self.background_img_dir = background_img_dir
        self.background_img_arr = sorted(glob.glob(f'{self.background_img_dir}/*.png')) + sorted(glob.glob(f'{self.background_img_dir}/*.jpg'))
        self.has_background = False
        self.hide_env = background_img_dir is not None

        self.smpl = SMPL(SMPL_MODEL_DIR, pose_type='body26fk', create_transl=False).to(device)
        faces = self.smpl.faces.copy()
        self.smpl_faces = faces = np.hstack([np.ones_like(faces[:, [0]]) * 3, faces])
        self.smpl_joint_parents = self.smpl.parents.cpu().numpy()
        self.last_fr = None
        self.align_freq = 150
        self.load_scene(data, coord)

    def get_aligned_orient_trans(self, pose_dict, exist_frames):
        orient_q = angle_axis_to_quaternion(pose_dict['smpl_orient_world'][exist_frames])
        trans = pose_dict['root_trans_world'][exist_frames]
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
        pose_dict['aligned_trans'] = torch.cat(pose_dict['aligned_trans'])
        pose_dict['smpl_orient_world'][exist_frames] = quaternion_to_angle_axis(pose_dict['aligned_orient_q'])
        pose_dict['root_trans_world'][exist_frames] = pose_dict['aligned_trans']

    def load_scene(self, data, coord):
        self.coord = coord 
        assert coord in {'cam', 'world', 'cam_in_world'}
        self.data = data
        self.scene_dict = data['person_data']
        self.gt = data['gt']
        self.gt_meta = data['gt_meta']

        self.focal_length = next(iter(self.scene_dict.values()))['cam_K'][0][[0, 1], [0, 1]]

        if len(self.gt) == 0:
            self.num_fr = self.scene_dict[0]['max_len']
            self.num_person = len(self.scene_dict)
        else:
            self.num_fr = self.gt[0]['pose'].shape[0]
            self.num_person = len(self.gt)
        self.unit = 0.001
        """ GT """
        suffix = '' if coord in {'world', 'cam_in_world'} else '_cam'
        for idx, pose_dict in self.gt.items():
            est_dict = self.scene_dict[idx]
            if self.use_est_traj:
                smpl_motion = self.smpl(
                    global_orient=torch.tensor(est_dict[f'smpl_orient_world']),
                    body_pose=torch.tensor(pose_dict['pose'][:, 3:]).float(),
                    betas=torch.tensor(est_dict['smpl_beta']),
                    root_trans = torch.tensor(est_dict[f'root_trans_world']),
                    root_scale = torch.tensor(est_dict['scale']) if est_dict['scale'] is not None else None,
                    return_full_pose=True,
                    orig_joints=True
                )
            else:
                pose_dict[f'smpl_orient_world'] = torch.tensor(pose_dict[f'pose{suffix}'][:, :3]).float()
                pose_dict[f'root_trans_world'] = torch.tensor(pose_dict[f'root_trans{suffix}']).float()
                if self.align_pose:
                    self.get_aligned_orient_trans(pose_dict, est_dict['exist_frames'])
                smpl_motion = self.smpl(
                    global_orient=pose_dict[f'smpl_orient_world'],
                    body_pose=torch.tensor(pose_dict['pose'][:, 3:]).float(),
                    betas=torch.tensor(pose_dict['shape']).float().repeat(pose_dict['pose'].shape[0], 1),
                    root_trans = pose_dict[f'root_trans_world'],
                    root_scale = None,
                    return_full_pose=True,
                    orig_joints=True
                )
            pose_dict['smpl_verts'] = smpl_motion.vertices.numpy()
            pose_dict['smpl_joints'] = smpl_motion.joints.numpy()
            if 'fr_start' not in pose_dict:
                pose_dict['fr_start'] = np.where(pose_dict['visible'])[0][0]

        """ Estimation """
        suffix = '' if coord == 'cam' else '_world'
        for pose_dict in self.scene_dict.values():
            if 'smpl_pose' in pose_dict:
                if not isinstance(pose_dict[f'smpl_orient{suffix}'], torch.Tensor):
                    pose_dict[f'smpl_orient{suffix}'] = torch.tensor(pose_dict[f'smpl_orient{suffix}'])
                    pose_dict[f'root_trans{suffix}'] = torch.tensor(pose_dict[f'root_trans{suffix}'])
                if self.align_pose and suffix == '_world':
                    self.get_aligned_orient_trans(pose_dict, pose_dict['exist_frames'])
                smpl_motion = self.smpl(
                    global_orient=pose_dict[f'smpl_orient{suffix}'],
                    body_pose=torch.tensor(pose_dict['smpl_pose']),
                    betas=torch.tensor(pose_dict['smpl_beta']),
                    root_trans = pose_dict[f'root_trans{suffix}'],
                    root_scale = torch.tensor(pose_dict['scale']) if pose_dict['scale'] is not None else None,
                    return_full_pose=True,
                    orig_joints=True
                )
                pose_dict['smpl_verts'] = smpl_motion.vertices.numpy()
                pose_dict['smpl_joints'] = smpl_motion.joints.numpy()
            if 'smpl_joint_pos' in pose_dict:
                orient = torch.tensor(pose_dict[f'smpl_orient{suffix}'])
                joints = torch.tensor(pose_dict['smpl_joint_pos'])
                trans = torch.tensor(pose_dict[f'root_trans{suffix}'])
                joints = torch.cat([torch.zeros_like(joints[..., :3]), joints], dim=-1).view(*joints.shape[:-1], -1, 3)
                orient_q = angle_axis_to_quaternion(orient).unsqueeze(-2).expand(joints.shape[:-1] + (4,))
                joints_world = quat_apply(orient_q, joints) + trans.unsqueeze(-2)
                pose_dict['smpl_joints'] = joints_world

        if 'exist_frames' in self.scene_dict[0]:
            self.init_est_root_pos = np.concatenate([x['smpl_joints'][x['exist_frames'], 0] for x in self.scene_dict.values()]).mean(axis=0)
        else:
            self.init_est_root_pos = np.concatenate([x['smpl_joints'][:, 0] for x in self.scene_dict.values()]).mean(axis=0)
        self.init_focal_point = self.init_est_root_pos

    def init_camera(self):
        if self.coord in {'cam', 'cam_in_world'}:
            self.pl.camera_position = 'zy'
            self.pl.camera.focal_point = (0, 0, 1)
            self.pl.camera.position = (0, 0, 0)
            self.pl.camera.up = (0, -1, 0)
            self.pl.camera.elevation = 0
            self.pl.camera.azimuth = 0
            self.set_camera_instrinsics(fx=self.focal_length[0], fy=self.focal_length[1])
        else:
            focal_point = self.init_focal_point
            if self.use_y_up_coord:
                focal_point[2] += 3.0
                self.pl.camera.position = (focal_point[0] + self.view_dist, focal_point[1] + 2, focal_point[2])
            else:
                self.pl.camera.position = (focal_point[0] + self.view_dist, focal_point[1], focal_point[2] + 2)
                cam_pose_inv = self.data['cam_pose_inv'][self.fr]
                cam_origin = cam_pose_inv[:3, 3].copy()
                cam_origin = (cam_origin - focal_point) * 1.5 + focal_point

                if self.render_cam_focus is not None:
                    self.pl.camera.focal_point = self.render_cam_focus
                    print('set camera focal:', self.pl.camera.focal_point)
                if self.render_cam_pos is not None:
                    self.pl.camera.position = self.render_cam_pos
                    print('set camera pos:', self.pl.camera.position)

            self.pl.camera.up = (0, 1, 0)

    def init_scene(self, init_args):
        if init_args is None:
            init_args = dict()
        super().init_scene(init_args)
        
        """ floor """
        whl = (20.0, 0.05, 20.0) if self.use_y_up_coord else (20.0, 20.0, 0.05)
        if self.coord == 'cam':
            center = np.array([0, whl[1] * 0.5 + 2, 7])
        else:
            center = self.init_focal_point
            if self.use_y_up_coord:
                center[1] = 0.0
            else:
                center[2] = -0.2

        if not self.hide_env:
            self.floor_mesh = pyvista.Cube(center, *whl)
            self.floor_mesh.t_coords *= 2 / self.floor_mesh.t_coords.max()
            tex = pyvista.numpy_to_texture(make_checker_board_texture('#81C6EB', '#D4F1F7'))
            self.pl.add_mesh(self.floor_mesh, texture=tex, ambient=0.2, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
        
        if self.coord == 'world' and self.show_camera:
            self.cam_sphere = pyvista.Sphere(radius=0.05, center=(0, 0, 2))
            self.cam_arrow_z = pyvista.Arrow(start=(0, 0, 2), direction=(0, 0, 1), scale=0.4)
            self.cam_arrow_y = pyvista.Arrow(start=(0, 0, 2), direction=(0, 0, 1), scale=0.4)
            self.cam_arrow_x = pyvista.Arrow(start=(0, 0, 2), direction=(0, 0, 1), scale=0.4)
            self.pl.add_mesh(self.cam_sphere, color='green', ambient=0.5, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
            self.pl.add_mesh(self.cam_arrow_z, color='yellow', ambient=0.5, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
            self.pl.add_mesh(self.cam_arrow_y, color='red', ambient=0.5, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)
            self.pl.add_mesh(self.cam_arrow_x, color='blue', ambient=0.5, diffuse=0.8, specular=0.8, specular_power=5, smooth_shading=True)

        
        vertices = self.gt[0]['smpl_verts'][0] if len(self.gt) > 0 else self.scene_dict[0]['smpl_verts'][0]
        colors = ['#33b400', '#8e95f2', 'orange', 'white', 'purple', 'cyan', 'blue', 'pink', 'red', 'green', 'yellow', 'black']
        self.smpl_actors = [SMPLActor(self.pl, vertices, self.smpl_faces, visible=False, color=color) for _, color in zip(range(self.num_person), colors)]
        self.smpl_gt_actors = [SMPLActor(self.pl, vertices, self.smpl_faces, visible=False) for _ in range(self.num_person)]
        self.skeleton_actors = [SkeletonActor(self.pl, self.smpl_joint_parents, bone_color='yellow', joint_color='green', visible=False) for _ in range(self.num_person)]
        self.skeleton_gt_actors = [SkeletonActor(self.pl, self.smpl_joint_parents, bone_color='red', joint_color='purple', visible=False) for _ in range(self.num_person)]
        self.smpl_actor_main = self.smpl_actors[0]
        
    def update_camera(self, interactive):
        if self.coord == 'cam_in_world':
            cam_pose_inv = self.data['cam_pose_inv'][self.fr]
            cam_origin = cam_pose_inv[:3, 3]
            view_vec = cam_pose_inv[:3, 2]
            up_vec = -cam_pose_inv[:3, 1]
            new_focal = cam_origin + view_vec
            self.pl.camera.up = up_vec.tolist()
            self.pl.camera.focal_point = new_focal.tolist()
            self.pl.camera.position = cam_origin.tolist()
        elif self.coord == 'cam':
            view_vec = np.asarray(self.pl.camera.position) - np.asarray(self.pl.camera.focal_point)
            new_focal = np.array(self.pl.camera.focal_point)
            new_pos = new_focal + view_vec
            self.pl.camera.up = (0, -1, 0)
            self.pl.camera.focal_point = new_focal.tolist()
            self.pl.camera.position = new_pos.tolist()
        else:
            if self.use_y_up_coord:
                self.pl.camera.up = (0, 1, 0)
            else:
                self.pl.camera.up = (0, 0, 1)

    def update_scene(self):
        super().update_scene()
        if self.verbose:
            print(self.fr)

        """ Background """
        if self.fr < len(self.background_img_arr):
            if self.interactive:
                if self.has_background:
                    self.pl.remove_background_image()
                self.pl.add_background_image(self.background_img_arr[self.fr])
                self.has_background = True
            else:
                self.background_img = self.background_img_arr[self.fr]


        """ Estimation """     
        if self.show_est_pose:
            i = 0
            j = 0
            for idx, pose_dict in self.scene_dict.items():
                actor = self.smpl_actors[i]
                sk_actor = self.skeleton_actors[j]
                if self.fr in pose_dict['frames']:
                    pind = pose_dict['frame2ind'][self.fr]
                    # smpl actor
                    if self.show_smpl and 'smpl_verts' in pose_dict:
                        if 'exist_frames' in pose_dict and not pose_dict['exist_frames'][self.fr]:
                            actor.set_visibility(False)
                        else:
                            actor.set_visibility(True)
                            verts_i = pose_dict['smpl_verts'][pind]
                            actor.update_verts(verts_i)
                            full_opacity = 0.7 if self.show_skeleton else 1.0
                            opacity = 0.4 if pose_dict['invis_frames'][self.fr] else full_opacity
                            actor.set_opacity(opacity)
                        i += 1
                    # skeleton actor
                    if self.show_skeleton and 'smpl_joints' in pose_dict:
                        sk_actor.set_visibility(True)
                        joints_i = pose_dict['smpl_joints'][pind]
                        sk_actor.update_joints(joints_i)
                        opacity = 0.4 if pose_dict['invis_frames'][self.fr] else 1.0
                        sk_actor.set_opacity(opacity)
                        j += 1
            for k in range(i, self.num_person):
                self.smpl_actors[k].set_visibility(False)
            for k in range(j, self.num_person):
                self.skeleton_actors[k].set_visibility(False)

        """ GT """
        if self.show_gt_pose:
            for i, actor in enumerate(self.smpl_gt_actors):
                pose_dict = self.gt[i]
                sk_actor = self.skeleton_gt_actors[i]
                # smpl actor
                if self.show_smpl:
                    actor.set_visibility(True)
                    verts_i = pose_dict['smpl_verts'][self.fr]
                    actor.update_verts(verts_i)
                    actor.set_opacity(0.5)
                else:
                    actor.set_visibility(False)
                # skeleton actor
                if self.show_skeleton:
                    sk_actor.set_visibility(True)
                    joints_i = pose_dict['smpl_joints'][self.fr]
                    sk_actor.update_joints(joints_i)
                    sk_actor.set_opacity(0.5)
                else:
                    sk_actor.set_visibility(False)

        if self.coord == 'world' and self.show_camera:
            cam_pose_inv = self.data['cam_pose_inv'][self.fr]
            new_sphere = pyvista.Sphere(radius=0.05, center=cam_pose_inv[:3, 3].tolist())
            new_arrow_z = pyvista.Arrow(start=cam_pose_inv[:3, 3].tolist(), direction=cam_pose_inv[:3, 2].tolist(), scale=0.4)
            new_arrow_y = pyvista.Arrow(start=cam_pose_inv[:3, 3].tolist(), direction=cam_pose_inv[:3, 1].tolist(), scale=0.4)
            new_arrow_x = pyvista.Arrow(start=cam_pose_inv[:3, 3].tolist(), direction=cam_pose_inv[:3, 0].tolist(), scale=0.4)
            self.cam_sphere.points[:] = new_sphere.points
            self.cam_arrow_z.points[:] = new_arrow_z.points
            self.cam_arrow_y.points[:] = new_arrow_y.points
            self.cam_arrow_x.points[:] = new_arrow_x.points

    def setup_key_callback(self):
        super().setup_key_callback()

        def go_to_frame():
            self.fr = 50
            if self.verbose:
                print(self.fr)
            self.paused = True
            self.update_scene()

        def print_camera():
            print(f"'cam_focus': {self.pl.camera.focal_point},")
            print(f"'cam_pos': {self.pl.camera.position}")

        def toggle_smpl():
            self.show_smpl = not self.show_smpl

        def toggle_skeleton():
            self.show_skeleton = not self.show_skeleton

        self.pl.add_key_event('t', print_camera)
        self.pl.add_key_event('z', go_to_frame)
        self.pl.add_key_event('j', toggle_smpl)
        self.pl.add_key_event('k', toggle_skeleton)

