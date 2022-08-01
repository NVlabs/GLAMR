import os, sys
sys.path.append(os.path.join(os.getcwd()))
import glob
import torch
import numpy as np
import pickle
import cv2 as cv
import shutil
from tqdm import tqdm
from collections import defaultdict
from lib.utils.joints import get_joints_info
from lib.models.smpl import SMPL, SMPL_MODEL_DIR
from lib.utils.vis import resize_bbox, images_to_video, draw_tracks
from lib.utils.torch_transform import quat_mul, quat_conjugate, angle_axis_to_quaternion, quaternion_to_angle_axis, rotation_matrix_to_quaternion, rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix


def convert_3dpw(data_path, output_path, split='all', j2d_conf_thresh=0.3):

    pose_path = f'{output_path}/pose'
    bbox_path = f'{output_path}/bbox'
    os.makedirs(pose_path, exist_ok=True)
    os.makedirs(bbox_path, exist_ok=True)

    # get a list of .pkl files in the directory
    seq_path = os.path.join(data_path, 'sequenceFiles', split)
    files = sorted(glob.glob(f'{seq_path}/*.pkl'))
    pose_type = 'body30'
    full_body_type = "body26fk"
    joints_info = get_joints_info(pose_type)

    smpl_male = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False, gender='male', pose_type=pose_type)
    smpl_female = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False, gender='female', pose_type=pose_type)
    smpl_full_m = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False, pose_type=full_body_type, gender="male")
    smpl_full_f = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False, pose_type=full_body_type, gender="female")

    src_joint_info = get_joints_info("coco")
    dst_joint_info = get_joints_info("body26fk")
    dst_dict = dict((v, k) for k, v in dst_joint_info.name.items())
    coco_to_body26fk = np.array([(dst_dict[v], k) for k, v in src_joint_info.name.items() if v in dst_dict.keys()])

    # go through all the .pkl Files
    for filename in tqdm(files):
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            seq_name = os.path.basename(filename).split('.')[0]
            imgs_path = os.path.join(data_path, 'imageFiles', seq_name)
            height, width, _ = cv.imread(os.path.join(imgs_path, 'image_00000.jpg')).shape
            K = data['cam_intrinsics']
            cam_pose = data['cam_poses']
            num_people = len(data['poses'])
            num_frames = len(data['img_frame_ids'])
            assert (data['poses'][0].shape[0] == num_frames)

            output_dict = defaultdict(dict)
            bbox_dict = defaultdict(dict)

            for p_id in range(num_people):
                output_dict[p_id] = defaultdict(list)
                pose = torch.from_numpy(data['poses'][p_id]).float()
                shape = torch.from_numpy(data['betas'][p_id][:10]).float().repeat(pose.size(0), 1)
                trans = torch.from_numpy(data['trans'][p_id]).float()
                j2d_coco = data['poses2d'][p_id].transpose(0, 2, 1)

                # ignore if 6 2D keypoints or valid camera lib
                valid_cam = data['campose_valid'][p_id].astype(bool)
                valid_pose = (j2d_coco[..., -1] > j2d_conf_thresh).astype(int).sum(axis=-1) >= 6
                visible_flag = (valid_pose & valid_cam).astype(int)

                gender = data['genders'][p_id]
                smpl_ = smpl_male if gender == 'm' else smpl_female
                output = smpl_(betas=shape, body_pose=pose[:, 3:], global_orient=pose[:, :3], transl=trans)
                smpl_full_ = smpl_full_m if gender == 'm' else smpl_full_f
                output_full = smpl_full_(betas=shape, body_pose=pose[:, 3:], global_orient=pose[:, :3], transl=trans)

                # verts = output.vertices
                # verts = torch.cat((verts, torch.ones(verts.shape[0], verts.shape[1], 1)), dim=2)
                # verts = np.matmul(verts, cam_pose.transpose(0, 2, 1))[:, :, :3]

                j3d = output.joints
                j3d = torch.cat((j3d, torch.ones(j3d.shape[0], j3d.shape[1], 1)), dim=2)
                j3d = np.matmul(j3d, cam_pose.transpose(0, 2, 1))[:, :, :3]

                j3d_full = output_full.joints
                j3d_full = torch.cat((j3d_full, torch.ones(j3d_full.shape[0], j3d_full.shape[1], 1)), dim=2)
                j3d_full = np.matmul(j3d_full, cam_pose.transpose(0, 2, 1))[:, :, :3]

                j2d_body26fk = np.zeros((j2d_coco.shape[0], 26, 3))
                j2d_body26fk[:, coco_to_body26fk[:, 0]] = j2d_coco[:, coco_to_body26fk[:, 1]]
                j2d = np.matmul(j3d, K.T)
                j2d = (j2d[..., :2] / j2d[..., -1:])

                root_trans = output.joints[:, 0]
                root_trans_cam = torch.cat((root_trans, torch.ones(root_trans.shape[0], 1)), dim=1)[:, None, :]
                root_trans_cam = np.matmul(root_trans_cam, cam_pose.transpose(0, 2, 1))[:, 0, :3]

                pose_cam = pose.clone()
                orient_qmat = angle_axis_to_rotation_matrix(pose_cam[..., :3])
                orient_qmat_cam = torch.matmul(torch.tensor(cam_pose[:, :3, :3]).float(), orient_qmat)
                orient_cam = rotation_matrix_to_angle_axis(orient_qmat_cam)
                pose_cam[..., :3] = orient_cam
                # output = smpl_(betas=shape, body_pose=pose_cam[:, 3:], global_orient=pose_cam[:, :3], root_trans=root_trans_cam)
                # j3d_cam = output.joints

                bbox = []
                for i in range(num_frames):
                    j2d_coco_i = j2d_coco[i]
                    coco_valid = j2d_coco_i[:,2] > 0.0
                    part = np.concatenate((j2d[i], j2d_coco_i[coco_valid,:2]), axis=0)
                    bbox_i = np.array([max(min(part[:, 0]), 0), max(min(part[:, 1]), 0),
                                       min(max(part[:, 0]), width), min(max(part[:, 1]), height)])
                    bbox.append(bbox_i)
                bbox = np.stack(bbox, axis=0)
                bbox = resize_bbox(bbox, 1.2)

                output_dict[p_id]['pose'] = data['poses'][p_id]
                output_dict[p_id]['shape'] = data['betas'][p_id][:10]
                output_dict[p_id]['trans'] = data['trans'][p_id]
                output_dict[p_id]['root_trans'] = root_trans.numpy()
                output_dict[p_id]['pose_cam'] = pose_cam.numpy()
                output_dict[p_id]['root_trans_cam'] = root_trans_cam.numpy()
                output_dict[p_id]['j3d'] = j3d.numpy()
                output_dict[p_id]['j2d'] = j2d.numpy()
                output_dict[p_id]['j3d_body26fk'] = j3d_full.numpy()
                output_dict[p_id]['j2d_body26fk'] = j2d_body26fk
                output_dict[p_id]['j2d_coco'] = j2d_coco
                output_dict[p_id]['visible'] = visible_flag
                output_dict[p_id]['bbox'] = bbox
                for key in output_dict[p_id].keys():
                    if key != 'visible':
                        output_dict[p_id][key] = output_dict[p_id][key].astype(np.float32)

                find = np.where(visible_flag)[0]
                bbox_dict[p_id]['id'] = p_id
                bbox_dict[p_id]['bbox'] = bbox
                bbox_dict[p_id]['exist'] = visible_flag
                bbox_dict[p_id]['score'] = visible_flag.astype(np.float32)
                bbox_dict[p_id]['start'] = find[0]
                bbox_dict[p_id]['end'] = find[-1]
                bbox_dict[p_id]['num_frames'] = visible_flag.sum()
                bbox_dict[p_id]['exist_frames'] = find
            
            out_dict = {
                'person_data': output_dict,
                'meta': {
                    'cam_pose': cam_pose,
                    'cam_K': K,
                    'campose_valid': data['campose_valid'],
                    'image_h': height,
                    'image_w': width
                }
            }
            pickle.dump(out_dict, open(f'{pose_path}/{seq_name}.pkl', 'wb'))
            pickle.dump(bbox_dict, open(f'{bbox_path}/{seq_name}.pkl', 'wb'))


def make_seq_videos(data_path, processed_path, video_path, split='all'):

    pose_path = f'{processed_path}/pose'
    bbox_path = f'{processed_path}/bbox'
    video_path = f'{processed_path}/videos'

    # get a list of .pkl files in the directory
    seq_path = os.path.join(data_path, 'sequenceFiles', split)
    files = sorted(glob.glob(f'{seq_path}/*.pkl'))
    seq_names = [os.path.splitext(os.path.basename(x))[0] for x in files]

    # go through all the .pkl Files
    for sind, seq_name in enumerate(seq_names):
        print(f'{sind}/{len(seq_names)} making video for {seq_name}')
        bbox_dict = pickle.load(open(f'{bbox_path}/{seq_name}.pkl', 'rb'))
        img_files = sorted(glob.glob(f'{data_path}/imageFiles/{seq_name}/*.jpg'))
        assert bbox_dict[0]['bbox'].shape[0] == len(img_files)
        frame_dir = f'tmp/3dpw_render/{seq_name}'
        vid_out_file = f'{video_path}/{seq_name}.mp4'
        os.makedirs(frame_dir, exist_ok=True)

        for find, img_path in enumerate(img_files):
            img = cv.imread(img_path)
            for idx, per_bbox_dict in bbox_dict.items():
                if find in per_bbox_dict['exist_frames']:
                    bbox = per_bbox_dict['bbox'][find]
                    img = draw_tracks(img, bbox, idx, per_bbox_dict['score'][find])
            cv.imwrite(f'{frame_dir}/{find:06d}.jpg', img)

        images_to_video(frame_dir, vid_out_file, fps=30, verbose=False)
        shutil.rmtree(frame_dir)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='datasets/3DPW')
    parser.add_argument('--output_path', default='datasets/3DPW/processed_v1')
    parser.add_argument('--video', action='store_true', default=False)
    args = parser.parse_args()

    if args.video:
        make_seq_videos(args.data_path, args.output_path, f'{args.output_path}/videos')
    else:
        convert_3dpw(args.data_path, args.output_path)

    print('done')
