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
from torchvision.models.detection import fasterrcnn_resnet50_fpn
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
parser.add_argument('--multi',
                    help='multi-person',
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

opt = parser.parse_args()


cfg_file = 'configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml'
cfg = update_config(cfg_file)

bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item*1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

transformation = SimpleTransform3DSMPL(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])

hybrik_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
hybrik_model.load_state_dict(torch.load(CKPT, map_location='cpu'), strict=False)
hybrik_model.cuda(opt.gpu)
hybrik_model.eval()

os.makedirs(os.path.join(opt.out_dir, 'res_images'), exist_ok=True)

files = os.listdir(f'{opt.img_folder}')
files.sort()

img_path_list = []
for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:
        img_path = os.path.join(opt.img_folder, file)
        img_path_list.append(img_path)

if opt.multi:

    from multi_person_tracker import MPT

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(opt.gpu))  
    else:
        device =  torch.device('cpu')

    # load multi-person tracking model
    mot = MPT(
        device=device,
        batch_size=32,
        display=False,
        detector_type='yolo',
        output_format='dict',
        yolo_img_size=416,
    )
    
    print('### Run Model...')
    tracking_results = mot(opt.img_folder)
    detection_all = defaultdict(dict)
    for person_id in tracking_results:
        frames_ids = tracking_results[person_id]['frames']
        for idx in range(len(frames_ids)):
            frames_id = frames_ids[idx]
            cx, cy, w, h = tracking_results[person_id]['bbox'][idx]
            x1, y1, x2, y2 = max(0, cx-w//2), max(0, cy-h//2), cx+w//2, cy+h//2
            detection_all[frames_id][person_id-1] = [x1, y1, x2, y2]

    out_dict = defaultdict(lambda: defaultdict(list))
    bbox_exist = defaultdict(list)
    bboxes = defaultdict(list)

    # initialize
    for person_id in tracking_results:
        bbox_exist[person_id-1] = [0 for _ in range(len(img_path_list))]

    for frame_idx in range(len(img_path_list)):

        img_path = img_path_list[frame_idx]
        dirname = os.path.dirname(img_path)
        basename = os.path.basename(img_path)

        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = input_image.copy()
        
        if frame_idx in detection_all:
            # For each detected person, starting from 0,1,...
            for idx in detection_all[frame_idx]:
                tight_bbox = detection_all[frame_idx][idx]
                bbox_exist[idx][frame_idx] = 1.0
                
                # Run HybrIK
                pose_input, bbox = transformation.test_transform(img_path, tight_bbox)
                pose_input = pose_input.to(opt.gpu)[None, :, :, :]
                pose_output = hybrik_model(pose_input)
                uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]

                # Visualization
                img_size = (image.shape[0], image.shape[1])
                focal = np.array([1000, 1000])
                bbox_xywh = xyxy2xywh(bbox)
                princpt = [bbox_xywh[0], bbox_xywh[1]]

                renderer = SMPLRenderer(faces=hybrik_model.smpl.faces,
                                        img_size=img_size, focal=focal,
                                        princpt=princpt)

                transl = pose_output.transl.detach().cpu().numpy().squeeze()
                transl[2] = transl[2] * 256 / bbox_xywh[2]

                image = vis_smpl_3d(
                    pose_output, image, cam_root=transl,
                    f=focal, c=princpt, renderer=renderer)

                # vis 2d
                pts = uv_29 * bbox_xywh[2]
                pts[:, 0] = pts[:, 0] + bbox_xywh[0]
                pts[:, 1] = pts[:, 1] + bbox_xywh[1]

                bboxes[idx].append(np.array(bbox_xywh))

                new_princpt = np.array([image.shape[1], image.shape[0]]) * 0.5
                transl[:2] += (np.array(princpt) - new_princpt) * transl[2] / np.array(focal) 
                princpt = new_princpt

                # save to dict
                K = np.eye(3)
                K[[0, 1], [0, 1]] = focal
                K[:2, 2] = princpt
                out_dict[idx]['smpl_pose_quat_wroot'].append(pose_output.pred_theta_mats[0].cpu().numpy().reshape(-1, 4))
                out_dict[idx]['smpl_beta'].append(pose_output.pred_shape[0].cpu().numpy())
                out_dict[idx]['root_trans'].append(transl)
                out_dict[idx]['kp_2d'].append(pts.cpu().numpy())
                out_dict[idx]['cam_K'].append(K.astype(np.float32))

        image_vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        res_path = os.path.join(opt.out_dir, 'res_images', f'{frame_idx:06d}.jpg')
        cv2.imwrite(res_path, image_vis)

    mot_bboxes = defaultdict(dict)
    for idx in bbox_exist:
        mot_bboxes[idx]['id'] = idx
        mot_bboxes[idx]['bbox'] = np.stack(bboxes[idx]),
        mot_bboxes[idx]['exist'] = np.array(bbox_exist[idx])
        
        find = np.where(mot_bboxes[idx]['exist'])[0]
        mot_bboxes[idx]['id'] = idx
        mot_bboxes[idx]['start'] = find[0]
        mot_bboxes[idx]['end'] = find[-1]
        mot_bboxes[idx]['num_frames'] = mot_bboxes[idx]['exist'].sum()
        mot_bboxes[idx]['exist_frames'] = find
        
    for idx, pose_dict in out_dict.items():
        for key in pose_dict.keys():
            pose_dict[key] = np.stack(pose_dict[key])
        pose_dict['frames'] = mot_bboxes[idx]['exist_frames']
        pose_dict['frame2ind'] = {f: i for i, f in enumerate(pose_dict['frames'])}
        pose_dict['bboxes_dict'] = mot_bboxes[idx]

else:
    # load detection model
    det_model = fasterrcnn_resnet50_fpn(pretrained=True)
    det_model.cuda(opt.gpu)
    det_model.eval()

    print('### Run Model...')

    prev_box = None

    out_dict = defaultdict(lambda: defaultdict(list))
    idx = 0   # single person id

    frame_idx = 0

    bbox_exist = []
    bboxes = []
    for img_path in tqdm(img_path_list):
        dirname = os.path.dirname(img_path)
        basename = os.path.basename(img_path)

        # Run Detection
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        det_input = det_transform(input_image).to(opt.gpu)
        det_output = det_model([det_input])[0]

        if prev_box is None:
            tight_bbox = get_one_box(det_output)  # xyxy
        else:
            tight_bbox = get_max_iou_box(det_output, prev_box)  # xyxy

        if tight_bbox is None:
            bbox_exist.append(0.0)
            continue
        else:
            bbox_exist.append(1.0)

        prev_box = tight_bbox

        # Run HybrIK
        pose_input, bbox = transformation.test_transform(img_path, tight_bbox)
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]
        pose_output = hybrik_model(pose_input)
        uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]

        # Visualization
        image = input_image.copy()
        img_size = (image.shape[0], image.shape[1])
        focal = np.array([1000, 1000])
        bbox_xywh = xyxy2xywh(bbox)
        princpt = [bbox_xywh[0], bbox_xywh[1]]

        renderer = SMPLRenderer(faces=hybrik_model.smpl.faces,
                                img_size=img_size, focal=focal,
                                princpt=princpt)

        transl = pose_output.transl.detach().cpu().numpy().squeeze()
        transl[2] = transl[2] * 256 / bbox_xywh[2]

        image_vis = vis_smpl_3d(
            pose_output, image, cam_root=transl,
            f=focal, c=princpt, renderer=renderer)

        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

        frame_idx += 1
        res_path = os.path.join(opt.out_dir, 'res_images', f'{frame_idx:06d}.jpg')
        cv2.imwrite(res_path, image_vis)

        # vis 2d
        pts = uv_29 * bbox_xywh[2]
        pts[:, 0] = pts[:, 0] + bbox_xywh[0]
        pts[:, 1] = pts[:, 1] + bbox_xywh[1]

        bboxes.append(np.array(bbox_xywh))

        new_princpt = np.array([image.shape[1], image.shape[0]]) * 0.5
        transl[:2] += (np.array(princpt) - new_princpt) * transl[2] / np.array(focal) 
        princpt = new_princpt

        # save to dict
        K = np.eye(3)
        K[[0, 1], [0, 1]] = focal
        K[:2, 2] = princpt
        out_dict[idx]['smpl_pose_quat_wroot'].append(pose_output.pred_theta_mats[0].cpu().numpy().reshape(-1, 4))
        out_dict[idx]['smpl_beta'].append(pose_output.pred_shape[0].cpu().numpy())
        out_dict[idx]['root_trans'].append(transl)
        out_dict[idx]['kp_2d'].append(pts.cpu().numpy())
        out_dict[idx]['cam_K'].append(K.astype(np.float32))
        
    mot_bboxes = {
        0: {
            'id': idx,
            'bbox': np.stack(bboxes),
            'exist': np.array(bbox_exist),
        }
    }
    
    find = np.where(mot_bboxes[idx]['exist'])[0]
    mot_bboxes[idx]['id'] = idx
    mot_bboxes[idx]['start'] = find[idx]
    mot_bboxes[idx]['end'] = find[-1]
    mot_bboxes[idx]['num_frames'] = mot_bboxes[idx]['exist'].sum()
    mot_bboxes[idx]['exist_frames'] = find
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
shutil.rmtree(f'{opt.out_dir}/res_images')
