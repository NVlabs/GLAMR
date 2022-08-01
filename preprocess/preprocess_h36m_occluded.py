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
import yaml
import subprocess
import cv2 as cv


from lib.utils.vis import images_to_video, draw_tracks, draw_keypoints


def gen_sinusoidal_traj(orig_traj, magnitude, period):
    t = np.arange(orig_traj.shape[0]).astype(float)
    offset = np.sin(t * (2 * np.pi / period))[:, None] * magnitude
    new_traj = orig_traj + offset
    return new_traj


def create_occluded_scene(pose_path, bbox_path, img_path, save_path, seq_name, specs):
    crop_h = specs['crop_h']
    crop_w = specs['crop_w']
    img_size = np.array([crop_w, crop_h]).astype(float)
    hsize = img_size * 0.5
    scene_dict = pickle.load(open(f'{pose_path}/{seq_name}.pkl', 'rb'))
    pose_dict = scene_dict['person_data']
    bbox_dict = pickle.load(open(f'{bbox_path}/{seq_name}.pkl', 'rb'))
    img_folder = f'{img_path}/{seq_name}'
    img_files = sorted(glob.glob(f'{img_folder}/*.jpg'))
    img_save_folder = f'{save_path}/images/{seq_name}'
    os.makedirs(img_save_folder, exist_ok=True)

    p_id = 0    # only a single person
    orig_img_h = scene_dict['meta']['image_h']
    orig_img_w = scene_dict['meta']['image_w']
    orig_img_size = np.array([orig_img_w, orig_img_h]).astype(float)

    """ Get Image Bbox """
    bbox = bbox_dict[p_id]['bbox']
    bbox_size = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    center = (bbox[:, :2] + bbox[:, 2:]) * 0.5

    new_img_orig = np.round(gen_sinusoidal_traj(center, np.array(specs['magnitude']), specs['period']))
    new_img_orig[:, :2] = np.maximum(hsize[None, :], new_img_orig[:, :2])
    new_img_orig[:, :2] = np.minimum((orig_img_size - hsize)[None, :], new_img_orig[:, :2])

    img_ul, img_br = new_img_orig - hsize, new_img_orig + hsize
    img_bbox = np.concatenate([img_ul, img_br], axis=-1)

    """ Pose """
    del pose_dict[p_id]['pose_cam']
    del pose_dict[p_id]['root_trans_cam']
    for key in pose_dict[p_id].keys():
        if 'j2d' in key:
            pose_dict[p_id][key][..., :2] = pose_dict[p_id][key][..., :2] - img_ul[:, None, :]
            visible = np.all((pose_dict[p_id][key][..., :2] >= 0) & (pose_dict[p_id][key][..., :2] <= img_size), axis=-1)
            visible = visible.astype(float)
            if pose_dict[p_id][key].shape[-1] == 3:
                pose_dict[p_id][key][..., 2] = visible
            pose_dict[p_id][key][..., :2] *= visible[..., None]
            if key == 'j2d_h36m':
                num_vis_joints = visible.sum(axis=-1)

    """ BBox """
    new_bbox = bbox.copy()
    new_bbox[:, :2] = np.maximum(img_ul, new_bbox[:, :2])
    new_bbox[:, 2:] = np.minimum(img_br, new_bbox[:, 2:])
    new_bbox_size = (new_bbox[:, 2] - new_bbox[:, 0]) * (new_bbox[:, 3] - new_bbox[:, 1])
    ratio = new_bbox_size / bbox_size
    visible_flag = (ratio >= specs['min_bbox_ratio']) & (num_vis_joints >= specs['min_vis_joints'])
    new_bbox -= np.tile(img_ul, (1, 2))
    new_bbox[visible_flag == 0.0] = 0.0
    find = np.where(visible_flag)[0]
    bbox_dict[p_id]['bbox'] = new_bbox
    bbox_dict[p_id]['exist'] = visible_flag
    bbox_dict[p_id]['score'] = visible_flag.astype(np.float32)
    bbox_dict[p_id]['start'] = find[0]
    bbox_dict[p_id]['end'] = find[-1]
    bbox_dict[p_id]['num_frames'] = visible_flag.sum()
    bbox_dict[p_id]['exist_frames'] = find
    pose_dict[p_id]['bbox'] = new_bbox
    pose_dict[p_id]['visible'] = visible_flag

    """ Meta """
    K = np.eye(3)
    K[[0, 1], [0, 1]] = max(crop_h, crop_w)
    K[:2, 2] = hsize
    scene_dict['meta']['cam_K'] = K
    scene_dict['meta']['cam_pose'] = np.tile(np.eye(4), (new_bbox.shape[0], 1, 1))
    scene_dict['meta']['image_h'] = crop_h
    scene_dict['meta']['image_w'] = crop_w

    pickle.dump(scene_dict, open(f'{save_path}/pose/{seq_name}.pkl', 'wb'))
    pickle.dump(bbox_dict, open(f'{save_path}/bbox/{seq_name}.pkl', 'wb'))

    """ Images """
    assert(len(img_files) == new_bbox.shape[0])
    for i, img_file in enumerate(img_files):
        img_name = osp.basename(img_file)
        img = cv.imread(img_file)
        ul = img_ul[i].astype(int)
        img_crop = img[ul[1]:ul[1] + crop_h, ul[0]:ul[0] + crop_w]
        cv.imwrite(f'{save_path}/images/{seq_name}/{img_name}', img_crop)


def make_video(save_path, seq_name):
    print(f'making video for {seq_name}')
    bbox_dict = pickle.load(open(f'{save_path}/bbox/{seq_name}.pkl', 'rb'))
    pose_dict = pickle.load(open(f'{save_path}/pose/{seq_name}.pkl', 'rb'))
    img_files = sorted(glob.glob(f'{save_path}/images/{seq_name}/*.jpg'))
    assert bbox_dict[0]['bbox'].shape[0] == len(img_files)
    frame_dir = f'tmp/h36m_render/{seq_name}'
    vid_out_file = f'{save_path}/video/{seq_name}.mp4'
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
    parser.add_argument('--img_path', default='datasets/H36M/images_25fps')
    parser.add_argument('--pose_path', default='datasets/H36M/processed_v1/pose')
    parser.add_argument('--bbox_path', default='datasets/H36M/processed_v1/bbox')
    parser.add_argument('--save_path', default='datasets/H36M/occluded_v2')
    parser.add_argument('--seq_range', default=None)
    parser.add_argument('--glob_pattern', default='*')
    parser.add_argument('--crop_w', type=int, default=300)
    parser.add_argument('--crop_h', type=int, default=600)
    parser.add_argument('--period', type=int, default=120)
    parser.add_argument('--magnitude', type=int, default=200)
    parser.add_argument('--min_vis_joints', type=int, default=15)
    parser.add_argument('--min_bbox_ratio', type=float, default=0.8)
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--video', action='store_true', default=False)
    args = parser.parse_args()
    

    bbox_path, img_path, pose_path = args.bbox_path, args.img_path, args.pose_path
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f'{save_path}/images', exist_ok=True)
    os.makedirs(f'{save_path}/bbox', exist_ok=True)
    os.makedirs(f'{save_path}/pose', exist_ok=True)
    yaml.safe_dump(args.__dict__, open(f'{save_path}/args.yml', 'w'))
    files = sorted(glob.glob(f'{bbox_path}/{args.glob_pattern}.pkl'))
    # files = sorted(glob.glob(f'{bbox_path}/s_09_*.pkl')) + sorted(glob.glob(f'{bbox_path}/s_11_*.pkl'))
    seq_names = [os.path.splitext(os.path.basename(x))[0] for x in files]

    occlude_specs = {
        'crop_h': args.crop_h,
        'crop_w': args.crop_w,
        'magnitude': [args.magnitude, 0],
        'period': args.period,
        'min_vis_joints': args.min_vis_joints,
        'min_bbox_ratio': args.min_bbox_ratio
    }

    if args.seq_range is not None:
        seq_range = [int(x) for x in args.seq_range.split('-')]
        seq_range = np.arange(seq_range[0], seq_range[1])
    else:
        seq_range = np.arange(len(seq_names))

    for sind in seq_range:
        seq_name = seq_names[sind]
        print(f'{sind}/{len(seq_names)} creating occluded data for {seq_name}')

        if args.video:
            make_video(save_path, seq_name)
        else:
            create_occluded_scene(pose_path, bbox_path, img_path, save_path, seq_name, occlude_specs)



