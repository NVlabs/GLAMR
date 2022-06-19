import yaml
import os
import glob
import numpy as np


class Config:

    def __init__(self, cfg_id, tmp=False):
        self.id = cfg_id
        cfg_path = 'motion_infiller/cfg_infer/**/%s.yml' % cfg_id
        files = glob.glob(cfg_path, recursive=True)
        assert(len(files) == 1)
        self.yml_dict = yml_dict = yaml.safe_load(open(files[0], 'r'))

        # data dir
        self.results_root_dir = os.path.expanduser(self.yml_dict['results_root_dir'])
        cfg_root_dir = '/tmp/motion_infiller' if tmp else f'{self.results_root_dir}'
        cfg_root_dir = os.path.expanduser(cfg_root_dir)
        self.cfg_dir = f'{cfg_root_dir}/{cfg_id}'
        self.log_dir = f'{self.cfg_dir}/logs'
        os.makedirs(self.log_dir, exist_ok=True)

        # misc
        self.seed = yml_dict.get('seed', 1)
        # model
        self.model_specs = yml_dict.get('model_specs', dict())

        # dataset
        self.amass_dir = yml_dict['amass_dir']
        self.seq_len = yml_dict.get('seq_len', 64)
        self.seq_sampling_method = yml_dict.get('seq_sampling_method', 'length')
        self.pose_gaussian_smooth = yml_dict.get('pose_gaussian_smooth', None)
        self.data_mask_methods = yml_dict.get('data_mask_methods', dict())

        # infer
        self.num_drop_fr = yml_dict.get('num_drop_fr', None)
        self.num_motion_samp = yml_dict.get('num_motion_samp', 3)
        self.multi_step_mfiller = yml_dict.get('multi_step_mfiller', True)
        self.multi_step_trajpred = yml_dict.get('multi_step_trajpred', True)
