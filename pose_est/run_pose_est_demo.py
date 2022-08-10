import os, sys
sys.path.append(os.path.join(os.getcwd()))
import os.path as osp
import subprocess
import glob
import numpy as np
import argparse
import yaml
from lib.utils.vis import video_to_images


def run_pose_est_on_video(video_file, output_dir, pose_est_model, cached_pose, gpu_index=0, multi=False):
    if pose_est_model == 'hybrik':
        if not (cached_pose and osp.exists(f'{output_dir}/pose.pkl')):
            image_folder = osp.join(output_dir, 'frames')
            video_to_images(video_file, image_folder, fps=30)
            conda_path = os.environ["CONDA_PREFIX"].split('/envs')[0]
            cmd = f'{conda_path}/envs/hybrik/bin/python ../pose_est/hybrik_demo/demo.py --img_folder {osp.abspath(image_folder)} --out_dir {osp.abspath(output_dir)} --gpu {gpu_index} --multi {1 if multi else 0}'
            subprocess.run(cmd.split(' '), cwd='./HybrIK')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default=None, help="path to video file or a directory that contains video files")
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--pose_est_model', default='hybrik')
    parser.add_argument('--pose_est_cfg', default=None)
    parser.add_argument('--seq_range', default=None)
    parser.add_argument('--glob_pattern', default='*')
    parser.add_argument('--render', action='store_true', default=True)
    parser.add_argument('--cached_pose', action='store_true', default=False)
    parser.add_argument('--cached_video', action='store_true', default=False)
    parser.add_argument('--merge_all_ids', action='store_true', default=True)
    parser.add_argument('--cleanup', action='store_true', default=True)
    args = parser.parse_args()

    video_path, output_path = args.video_path, args.output_path
    os.makedirs(output_path, exist_ok=True)
    yaml.safe_dump(args.__dict__, open(f'{output_path}/args.yml', 'w'))

    """ single file """
    if osp.isfile(video_path):
        if osp.splitext(video_path)[1] != '.mp4':
            raise ValueError('Unsupported video file format!')
        print(f'estimating pose for {video_path}')
        output_dir = osp.join(output_path, osp.splitext(osp.basename(video_path))[0])

        run_pose_est_on_video(video_path, output_dir, args.pose_est_model, args.cached_pose)

    else:
        files = sorted(glob.glob(f'{video_path}/{args.glob_pattern}.mp4'))
        seq_names = [os.path.splitext(os.path.basename(x))[0] for x in files]
        if args.seq_range is not None:
            seq_range = [int(x) for x in args.seq_range.split('-')]
            seq_range = np.arange(seq_range[0], seq_range[1])
        else:
            seq_range = np.arange(len(seq_names))

        for sind in seq_range:
            seq_name = seq_names[sind]
            print(f'{sind}/{len(seq_names)} estimating pose for {seq_name}')
            seq_video_path = f'{video_path}/{seq_name}.mp4'
            output_dir = f'{output_path}/{seq_name}'

            run_pose_est_on_video(seq_video_path, output_dir, args.pose_est_model, args.cached_pose)
