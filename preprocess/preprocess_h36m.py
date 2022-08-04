import os, sys
sys.path.append(os.path.join(os.getcwd()))
import os.path as osp
import shutil
import torch
import numpy as np
import json
import glob
import sys
import pickle
import joblib
import subprocess
import cv2 as cv


from collections import defaultdict
from scipy.interpolate import interp1d
from lib.models.smpl import SMPL, SMPL_MODEL_DIR
from lib.utils.vis import resize_bbox, images_to_video, draw_tracks, draw_keypoints
from lib.utils.torch_transform import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix


genders = {
    1: 'f',
    5: 'f',
    6: 'm',
    7: 'f',
    8: 'm',
    9: 'm',
    11: 'm'
}


def get_ordered_action_names(action_names, subject_id):
    new_action_names = [None] * 30
    for i, aname in enumerate(action_names):
        if 'Directions' in aname:
            index = 0 * 2 + (i % 2)
        elif 'Discussion' in aname:
            index = 1 * 2 + (i % 2)
        elif 'Eating' in aname:
            index = 2 * 2 + (i % 2)
        elif 'Greeting' in aname:
            index = 3 * 2 + (i % 2)
        elif 'Phoning' in aname:
            index = 4 * 2 + ((i + (1 if subject_id in {11} else 0)) % 2)
        elif 'Posing' in aname:
            index = 5 * 2 + (i % 2)
        elif 'Purchases' in aname:
            index = 6 * 2 + (i % 2)
        elif 'Down' in aname:
            index = 8 * 2 + ((i + (1 if subject_id in {5, 7, 8, 9, 11} else 0)) % 2)
        elif 'Sitting' in aname:
            index = 7 * 2 + (i % 2)
        elif 'Smoking' in aname:
            index = 9 * 2 + (i % 2)
        elif 'Photo' in aname:
            index = 10 * 2 + ((i + (1 if subject_id in {5, 6, 7} else 0)) % 2)
        elif 'Waiting' in aname:
            index = 11 * 2 + (i % 2)
        elif 'Dog' in aname:
            index = 13 * 2 + (i % 2)
        elif 'Together' in aname:
            index = 14 * 2 + (i % 2)
        elif 'Walking' in aname:
            index = 12 * 2 + (i % 2)
        new_action_names[index] = aname
    return new_action_names


def convert_h36m(h36m_folder, h36m_out_folder, subject_id, smpl_fit_data=None, use_smpl_fit=True, convert_img=False, cached=False):
    print(f'convert H36M for subject {subject_id}')
    gender = genders[subject_id]

    pose_path = f'{h36m_out_folder}/pose'
    bbox_path = f'{h36m_out_folder}/bbox'
    os.makedirs(pose_path, exist_ok=True)
    os.makedirs(bbox_path, exist_ok=True)

    smpl_file = f'{h36m_folder}/smpl_param/Human36M_subject{subject_id}_smpl_param.json'
    cam_file = f'{h36m_folder}/annotations/Human36M_subject{subject_id}_camera.json'
    joint_file = f'{h36m_folder}/annotations/Human36M_subject{subject_id}_joint_3d.json'

    with open(cam_file) as f:
        orig_cam_dict = json.load(f)
    cam_dict = dict()
    for cam_id in range(1, 5):
        cam = orig_cam_dict[str(cam_id)]
        for key in cam.keys():
            cam[key] = np.array(cam[key])
        cam['t'] *= 0.001
        cam_dict[cam_id] = cam

    if not use_smpl_fit:
        with open(smpl_file) as f:
            smpl_param = json.load(f)

    with open(joint_file) as f:
        joint = json.load(f)

    pose_type = 'body30'
    full_body_type = "body26fk"
    smpl_male = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False, gender='male', pose_type=pose_type)
    smpl_female = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False, gender='female', pose_type=pose_type)
    smpl_full_m = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False, pose_type=full_body_type, gender="male")
    smpl_full_f = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False, pose_type=full_body_type, gender="female")

    if use_smpl_fit:
        smpl_fit_seq_names = [x for x in smpl_fit_data.keys() if f'S{subject_id}-' in x]
        smpl_fit_seq_names = get_ordered_action_names(smpl_fit_seq_names, subject_id)
        seq_ind = 0

    for action_id in range(2, 17):
        for sub_action in range(1, 3):
            """ get sequence data first """
            seq = defaultdict(list)
            seq_name = f's_{subject_id:02d}_act_{action_id:02d}_subact_{sub_action:02d}'
            if convert_img:
                for cam_id in range(1, 5):
                    if f'{seq_name}_ca_{cam_id:02d}' == 's_11_act_02_subact_02_ca_01':
                        continue
                    img_folder = f'{h36m_folder}/images/{seq_name}_ca_{cam_id:02d}'
                    out_img_folder = f'{h36m_folder}/images_25fps/{seq_name}_ca_{cam_id:02d}'
                    img_list = sorted(glob.glob(f'{img_folder}/*.jpg'))
                    os.makedirs(out_img_folder, exist_ok=True)
                    for img in img_list[::2]:
                        shutil.copyfile(img, f'{out_img_folder}/{osp.basename(img)}')
                img_list = img_list[::2]
            else:
                img_list = glob.glob(f'{h36m_folder}/images_25fps/{seq_name}_ca_02/*.jpg')

            # get gt
            gt_jpos_dict = joint[str(action_id)][str(sub_action)]
            orig_num_fr = len(gt_jpos_dict)
            num_fr = len(img_list)
            for i, key in enumerate(gt_jpos_dict.keys()):
                assert str(i) == key
                seq['gt_jpos'].append(np.array(gt_jpos_dict[key]) * 0.001)
            seq['gt_jpos'] = np.stack(seq['gt_jpos'])[::2]      # 25fps
            assert(len(seq['gt_jpos'] == num_fr))

            if use_smpl_fit:
                smpl_fit_seq = smpl_fit_data[smpl_fit_seq_names[seq_ind]]
                pose = smpl_fit_seq['pose'][::2]
                print(f'{seq_ind} {smpl_fit_seq_names[seq_ind]} {pose.shape[0]} {num_fr}')
                assert(num_fr == pose.shape[0])
                for key in ['trans', 'pose', 'shape']:
                    seq[key] = smpl_fit_seq[key][::2]
                seq_ind += 1
            else:
                smpl_param_dict = smpl_param[str(action_id)][str(sub_action)]
                frames = np.array(sorted([int(x) for x in smpl_param_dict.keys()]))
                diff = np.diff(frames)
                ind = np.where(diff != 5)[0]
                print(f'irregular frames:', [(x, y) for x, y in zip(ind, diff[ind])])
                for i in sorted([int(x) for x in smpl_param_dict.keys()]):
                    key = str(i)
                    seq['fitted_jpos'].append(np.array(smpl_param_dict[key]['fitted_3d_pose']) * 0.001)
                    seq['trans'].append(np.array(smpl_param_dict[key]['trans'][0]))
                    seq['pose'].append(np.array(smpl_param_dict[key]['pose']))
                    seq['shape'].append(np.array(smpl_param_dict[key]['shape']))

                for key in ['fitted_jpos', 'trans', 'pose', 'shape']:
                    seq[key] = np.stack(seq[key])
                    f = interp1d(frames.astype(np.float32), seq[key], axis=0, assume_sorted=True, fill_value="extrapolate")
                    seq[key] = f(np.arange(orig_num_fr, step=2, dtype=np.float32))

            """ convert to our format """
            p_id = 0
            output_dict = {p_id: defaultdict(list)}
            bbox_dict = {p_id: defaultdict(list)}
            pose = torch.from_numpy(seq['pose']).float()
            shape = torch.from_numpy(seq['shape']).float()
            trans = torch.from_numpy(seq['trans']).float()

            for cam_id in range(1, 5):
                full_seq_name = f'{seq_name}_ca_{cam_id:02d}'
                if full_seq_name == 's_11_act_02_subact_02_ca_01':
                    continue
                if cached and osp.exists(f'{pose_path}/{full_seq_name}.pkl') and osp.exists(f'{bbox_path}/{full_seq_name}.pkl'):
                    continue
                print(f'converting {full_seq_name}')
                height, width, _ = cv.imread(f'{h36m_folder}/images_25fps/{full_seq_name}/{full_seq_name}_000001.jpg').shape
                cam_pose = np.eye(4)
                cam_pose[:3, :3] = cam_dict[cam_id]['R']
                cam_pose[:3, 3] = cam_dict[cam_id]['t']
                K = np.eye(3)
                K[[0, 1], [0, 1]] = cam_dict[cam_id]['f']
                K[:2, 2] = cam_dict[cam_id]['c']
                
                visible_flag = np.ones(num_fr)

                smpl_ = smpl_male if gender == 'm' else smpl_female
                smpl_full_ = smpl_full_m if gender == 'm' else smpl_full_f
                if use_smpl_fit:
                    output = smpl_(betas=shape, body_pose=pose[:, 3:], global_orient=pose[:, :3], root_trans=trans)
                    output_full = smpl_full_(betas=shape, body_pose=pose[:, 3:], global_orient=pose[:, :3], root_trans=trans)
                else:
                    output = smpl_(betas=shape, body_pose=pose[:, 3:], global_orient=pose[:, :3], transl=trans)
                    output_full = smpl_full_(betas=shape, body_pose=pose[:, 3:], global_orient=pose[:, :3], transl=trans)

                j3d = output.joints
                j3d = torch.cat((j3d, torch.ones(j3d.shape[0], j3d.shape[1], 1)), dim=2)
                j3d = np.matmul(j3d, cam_pose.T)[:, :, :3]
                j2d = np.matmul(j3d, K.T)
                j2d = (j2d[..., :2] / j2d[..., -1:])

                j3d_full = output_full.joints
                j3d_full = torch.cat((j3d_full, torch.ones(j3d_full.shape[0], j3d_full.shape[1], 1)), dim=2)
                j3d_full = np.matmul(j3d_full, cam_pose.T)[:, :, :3]
                j2d_body26fk = np.matmul(j3d_full, K.T)
                j2d_body26fk = (j2d_body26fk[..., :2] / j2d_body26fk[..., -1:])
                j2d_body26fk = torch.cat((j2d_body26fk, torch.ones(*j2d_body26fk.shape[:2], 1)), dim=-1)

                j3d_h36m_world = torch.from_numpy(seq['gt_jpos']).float()
                j3d_h36m = j3d_h36m_world.clone()
                j3d_h36m = torch.cat((j3d_h36m, torch.ones(j3d_h36m.shape[0], j3d_h36m.shape[1], 1)), dim=2)
                j3d_h36m = np.matmul(j3d_h36m, cam_pose.T)[:, :, :3]
                j2d_h36m = np.matmul(j3d_h36m, K.T)
                j2d_h36m = (j2d_h36m[..., :2] / j2d_h36m[..., -1:])

                root_trans = output.joints[:, 0]
                root_trans_cam = torch.cat((root_trans, torch.ones(root_trans.shape[0], 1)), dim=1)[:, None, :]
                root_trans_cam = np.matmul(root_trans_cam, cam_pose.T)[:, 0, :3]

                pose_cam = pose.clone()
                orient_qmat = angle_axis_to_rotation_matrix(pose_cam[..., :3])
                orient_qmat_cam = torch.matmul(torch.tensor(cam_pose[:3, :3]).float(), orient_qmat)
                orient_cam = rotation_matrix_to_angle_axis(orient_qmat_cam)
                pose_cam[..., :3] = orient_cam

                bbox = []
                for i in range(num_fr):
                    part = np.concatenate((j2d[i], j2d_h36m[i]), axis=0)
                    bbox_i = np.array([max(min(part[:, 0]), 0), max(min(part[:, 1]), 0),
                                        min(max(part[:, 0]), width), min(max(part[:, 1]), height)])
                    bbox.append(bbox_i)
                bbox = np.stack(bbox, axis=0)
                bbox = resize_bbox(bbox, 1.2)

                output_dict[p_id]['pose'] = pose.numpy()
                output_dict[p_id]['shape'] = shape.mean(dim=0).numpy()
                output_dict[p_id]['trans'] = trans.numpy()
                output_dict[p_id]['root_trans'] = root_trans.numpy()
                output_dict[p_id]['pose_cam'] = pose_cam.numpy()
                output_dict[p_id]['root_trans_cam'] = root_trans_cam.numpy()
                output_dict[p_id]['j3d'] = j3d.numpy()
                output_dict[p_id]['j3d_h36m'] = j3d_h36m.numpy()
                output_dict[p_id]['j3d_h36m_world'] = j3d_h36m_world.numpy()
                output_dict[p_id]['j2d'] = j2d.numpy()
                output_dict[p_id]['j3d_body26fk'] = j3d_full.numpy()
                output_dict[p_id]['j2d_body26fk'] = j2d_body26fk.numpy()
                output_dict[p_id]['j2d_h36m'] = j2d_h36m.numpy()
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
                        'image_h': height,
                        'image_w': width
                    }
                }
                pickle.dump(out_dict, open(f'{pose_path}/{full_seq_name}.pkl', 'wb'))
                pickle.dump(bbox_dict, open(f'{bbox_path}/{full_seq_name}.pkl', 'wb'))


def make_seq_videos(image_path, pose_path, bbox_path, video_path):

    # get a list of .pkl files in the directory
    files = sorted(glob.glob(f'{pose_path}/s_*_ca_01.pkl'))
    seq_names = [os.path.splitext(os.path.basename(x))[0] for x in files]

    # go through all the .pkl Files
    for sind, seq_name in enumerate(seq_names[:10]):
        print(f'{sind}/{len(seq_names)} making video for {seq_name}')
        bbox_dict = pickle.load(open(f'{bbox_path}/{seq_name}.pkl', 'rb'))
        pose_dict = pickle.load(open(f'{pose_path}/{seq_name}.pkl', 'rb'))
        img_files = sorted(glob.glob(f'{image_path}/{seq_name}/*.jpg'))
        assert bbox_dict[0]['bbox'].shape[0] == len(img_files)
        frame_dir = f'tmp/h36m_render/{seq_name}'
        vid_out_file = f'{video_path}/{seq_name}.mp4'
        os.makedirs(frame_dir, exist_ok=True)

        for find, img_path in enumerate(img_files):
            img = cv.imread(img_path)
            for idx, per_bbox_dict in bbox_dict.items():
                if find in per_bbox_dict['exist_frames']:
                    bbox = per_bbox_dict['bbox'][find]
                    kp = pose_dict['person_data'][idx]['j2d_body26fk'][find]
                    img = draw_tracks(img, bbox, idx, per_bbox_dict['score'][find])
                    img = draw_keypoints(img, kp[:, :2], kp[:, 2])
            cv.imwrite(f'{frame_dir}/{find:06d}.jpg', img)

        images_to_video(frame_dir, vid_out_file, fps=30, verbose=False)
        shutil.rmtree(frame_dir)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--h36m_folder', default="datasets/H36M")
    parser.add_argument('--h36m_out_folder', default="datasets/H36M/processed_v1")
    parser.add_argument('--subjects', default="1,5,6,7,8,9,11")
    parser.add_argument('--convert_image', action='store_true', default=True)
    parser.add_argument('--use_smpl_fit', action='store_true', default=True)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--video', action='store_true', default=False)
    args = parser.parse_args()

    subjects = [int(x) for x in args.subjects.split(',')]

    if args.video:
        make_seq_videos(f'{args.h36m_folder}/images_25fps', f'{args.h36m_out_folder}/pose', f'{args.h36m_out_folder}/bbox', f'{args.h36m_out_folder}/video')
    else:
        smpl_fit_data = joblib.load(f'{args.h36m_folder}/smpl_fit/h36m_train_60_fitted_grad.pkl')
        test_smpl_fit_data = joblib.load(f'{args.h36m_folder}/smpl_fit/h36m_test_60_fitted_grad.pkl')
        smpl_fit_data.update(test_smpl_fit_data)
        for sub in subjects:
            convert_h36m(args.h36m_folder, args.h36m_out_folder, sub, smpl_fit_data, args.use_smpl_fit, args.convert_image, args.cached)
