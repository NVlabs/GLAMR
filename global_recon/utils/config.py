import shutil
import yaml
import os
import glob
import numpy as np


def sci_to_int(val):
    return int(float(val))


class Config:

    def __init__(self, cfg_id, out_dir=None, tmp=False):
        self.id = cfg_id
        cfg_path = 'global_recon/cfg/**/%s.yml' % cfg_id
        files = glob.glob(cfg_path, recursive=True)
        assert(len(files) == 1)
        self.yml_file = files[0]
        self.yml_dict = yml_dict = yaml.safe_load(open(self.yml_file, 'r'))

        # result dir
        if out_dir is None:
            self.results_root_dir = os.path.expanduser(self.yml_dict['results_root_dir'])
            cfg_root_dir = 'tmp/global_recon' if tmp else f'{self.results_root_dir}'
            cfg_root_dir = os.path.expanduser(cfg_root_dir)
            self.cfg_dir = f'{cfg_root_dir}/{cfg_id}'
        else:
            self.cfg_dir = out_dir
        self.log_dir = f'{self.cfg_dir}/logs'
        os.makedirs(self.log_dir, exist_ok=True)

        # misc
        self.seed = yml_dict.get('seed', 1)
        # global recon model
        self.grecon_model_name = yml_dict['grecon_model_name']
        self.grecon_model_specs = yml_dict.get('grecon_model_specs', dict())
        self.opt_stage_specs = yml_dict.get('opt_stage_specs', dict())
        # dataset
        self.dataset = yml_dict.get('dataset', None)
        self.img_path = yml_dict.get('img_path', None)
        self.video_path = yml_dict.get('video_path', None)
        self.pose_path = yml_dict.get('pose_path', None)
        self.bbox_path = yml_dict.get('bbox_path', None)
        self.pose_est_path = yml_dict.get('pose_est_path', None)
        self.cam_est_path = yml_dict.get('cam_est_path', None)

    def save_yml_file(self, out_path=None):
        if out_path is None:
            out_path = f'{self.cfg_dir}/cfg.yml'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        shutil.copyfile(self.yml_file, out_path)


            