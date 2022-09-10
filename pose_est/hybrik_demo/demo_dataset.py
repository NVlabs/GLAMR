"""Image demo script."""
import argparse
import os
import sys
import os.path as osp
import subprocess
sys.path.append('./')

import cv2
import numpy as np
import torch
import shutil
import pickle
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPL
from hybrik.utils.render import SMPLRenderer
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d, vis_smpl_3d
from torchvision import transforms as T
from tqdm import tqdm
from collections import defaultdict


torch.set_grad_enabled(False)
det_transform = T.Compose([T.ToTensor()])

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def images_to_video(img_dir, out_path, img_fmt="%06d.jpg", fps=30, crf=25, verbose=True):
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    FFMPEG_PATH = '/usr/bin/ffmpeg' if osp.exists('/usr/bin/ffmpeg') else 'ffmpeg'
    cmd = [FFMPEG_PATH, '-y', '-r', f'{fps}', '-f', 'image2', '-start_number', '0',
            '-i', f'{img_dir}/{img_fmt}', '-vcodec', 'libx264', '-crf', f'{crf}', '-pix_fmt', 'yuv420p', out_path]
    if not verbose:
        cmd += ['-hide_banner', '-loglevel', 'error']
    subprocess.run(cmd)


parser = argparse.ArgumentParser(description='HybrIK Demo')
CKPT = 'pretrained_w_cam.pth'

parser.add_argument('--gpu',
                    help='gpu',
                    default=0,
                    type=int)
parser.add_argument('--img_folder',
                    help='image name',
                    default='',
                    type=str)
parser.add_argument('--out_dir',
                    help='output folder',
                    default='',
                    type=str)
parser.add_argument('--bbox_file',
                    help='Multi-object tracking (MOT) bbox file',
                    default='',
                    type=str)

opt = parser.parse_args()


cfg_file = 'configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml'
cfg = update_config(cfg_file)

dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': (2.2, 2.2, 2.2)
})

transformation = SimpleTransform3DSMPL(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=(2.2, 2,2, 2.2),
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])


hybrik_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
hybrik_model.load_state_dict(torch.load(CKPT, map_location='cpu'), strict=False)

hybrik_model.cuda(opt.gpu)
hybrik_model.eval()

os.makedirs(os.path.join(opt.out_dir, 'res_images'), exist_ok=True)
os.makedirs(os.path.join(opt.out_dir, 'res_2d_images'), exist_ok=True)

files = os.listdir(f'{opt.img_folder}')
files.sort()

img_path_list = []

for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:

        img_path = os.path.join(opt.img_folder, file)
        img_path_list.append(img_path)

mot_bboxes = pickle.load(open(opt.bbox_file, 'rb'))

prev_box = None

print('### Run Model...')

out_dict = defaultdict(lambda: defaultdict(list))
idx = 0   # current not supporting multiple person in this demo

frame_idx = 0

for fr, img_path in enumerate(tqdm(img_path_list)):
    dirname = os.path.dirname(img_path)
    basename = os.path.basename(img_path)
    # Run Detection
    input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    image_vis = input_image.copy()
    image_vis2d = input_image.copy()

    for idx, bbox_dict in mot_bboxes.items():

        if bbox_dict['exist'][fr] == 0:
            continue

        tight_bbox = bbox_dict['bbox'][fr]

        # Run HybrIK
        pose_input, bbox = transformation.test_transform(img_path, tight_bbox)
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]
        pose_output = hybrik_model(pose_input)
        uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]

        # Visualization
        img_size = (image_vis.shape[0], image_vis.shape[1])
        focal = np.array([1000, 1000])
        bbox_xywh = xyxy2xywh(bbox)
        princpt = [bbox_xywh[0], bbox_xywh[1]]

        renderer = SMPLRenderer(faces=hybrik_model.smpl.faces,
                                img_size=img_size, focal=focal,
                                princpt=princpt)

        transl = pose_output.transl.detach().cpu().numpy().squeeze()
        transl[2] = transl[2] * 256 / bbox_xywh[2]

        new_princpt = np.array([input_image.shape[1], input_image.shape[0]]) * 0.5
        transl[:2] += (np.array(princpt) - new_princpt) * transl[2] / np.array(focal) 
        princpt = new_princpt

        image_vis = vis_smpl_3d(
            pose_output, image_vis, cam_root=transl,
            f=focal, c=princpt, renderer=renderer)

        # vis 2d
        pts = uv_29 * bbox_xywh[2]
        pts[:, 0] = pts[:, 0] + bbox_xywh[0]
        pts[:, 1] = pts[:, 1] + bbox_xywh[1]
        image_vis2d = vis_2d(image_vis2d, tight_bbox, pts)

        # save to dict
        K = np.eye(3)
        K[[0, 1], [0, 1]] = focal
        K[:2, 2] = princpt
        out_dict[idx]['smpl_pose_quat_wroot'].append(pose_output.pred_theta_mats[0].cpu().numpy().reshape(-1, 4))
        out_dict[idx]['smpl_beta'].append(pose_output.pred_shape[0].cpu().numpy())
        out_dict[idx]['root_trans'].append(transl)
        out_dict[idx]['kp_2d'].append(pts.cpu().numpy())
        out_dict[idx]['cam_K'].append(K.astype(np.float32))
    
    frame_idx += 1
    image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
    res_path = os.path.join(opt.out_dir, 'res_images', f'{frame_idx:06d}.jpg')
    cv2.imwrite(res_path, image_vis)

    # image_vis2d = cv2.cvtColor(image_vis2d, cv2.COLOR_RGB2BGR)
    # res_path = os.path.join(opt.out_dir, 'res_2d_images', f'{frame_idx:06d}.jpg')
    # cv2.imwrite(res_path, image_vis2d)


for idx, pose_dict in out_dict.items():
    for key in pose_dict.keys():
        pose_dict[key] = np.stack(pose_dict[key])
    pose_dict['frames'] = mot_bboxes[idx]['exist_frames']
    pose_dict['frame2ind'] = {f: i for i, f in enumerate(pose_dict['frames'])}
    pose_dict['bboxes_dict'] = mot_bboxes[idx]

new_dict = dict()
for k, v in out_dict.items():
    new_dict[k] = dict()
    for ck, cv in v.items():
        new_dict[k][ck] = cv
pickle.dump(new_dict, open(f'{opt.out_dir}/pose.pkl', 'wb'))  

images_to_video(f'{opt.out_dir}/res_images', f'{opt.out_dir}/render.mp4', img_fmt='%06d.jpg')
# images_to_video(f'{opt.out_dir}/res_2d_images', f'{opt.out_dir}/render_kp2d.mp4', img_fmt='%06d.jpg')

shutil.rmtree(f'{opt.out_dir}/res_images')
# shutil.rmtree(f'{opt.out_dir}/res_2d_images')