import yaml
import os
import glob
import numpy as np


def sci_to_int(val):
    return int(float(val))


class Config:

    def __init__(self, cfg_id, tmp=False, training=True):
        self.id = cfg_id
        self.training = training
        cfg_path = 'motion_infiller/cfg/**/%s.yml' % cfg_id
        files = glob.glob(cfg_path, recursive=True)
        assert(len(files) == 1)
        self.yml_dict = yml_dict = yaml.safe_load(open(files[0], 'r'))

        # data dir
        self.results_root_dir = os.path.expanduser(self.yml_dict['results_root_dir'])
        cfg_root_dir = '/tmp/motion_infiller' if tmp else f'{self.results_root_dir}'
        cfg_root_dir = os.path.expanduser(cfg_root_dir)
        self.cfg_dir = f'{cfg_root_dir}/{cfg_id}'
        os.makedirs(self.cfg_dir, exist_ok=True)

        # misc
        self.seed = yml_dict.get('seed', 1)
        # model
        self.model_name = yml_dict['model_name']
        self.model_specs = yml_dict.get('model_specs', dict())
        self.loss_cfg = yml_dict.get('loss_cfg', dict())
        self.lr = yml_dict.get('lr', 1e-3)
        self.lr_scheduler = yml_dict.get('lr_scheduler', None)
        self.gradient_clip_val = yml_dict.get('gradient_clip_val', 0.0)

        # dataset
        self.amass_dir = yml_dict['amass_dir']
        self.max_epochs = sci_to_int(yml_dict.get('max_epochs', 20))
        self.save_n_epochs = sci_to_int(yml_dict.get('save_n_epochs', 1))
        self.train_ntime_per_epoch = sci_to_int(yml_dict.get('train_ntime_per_epoch', 1e3))
        self.val_ntime_per_epoch = sci_to_int(yml_dict.get('val_ntime_per_epoch', 1e3))
        self.seq_len = yml_dict.get('seq_len', 64)
        self.seq_sampling_method = yml_dict.get('seq_sampling_method', 'uniform')
        self.pose_gaussian_smooth = yml_dict.get('pose_gaussian_smooth', None)
        self.batch_size = sci_to_int(yml_dict.get('batch_size', 64))
        self.data_mask_methods = yml_dict.get('data_mask_methods', dict())

        if not self.training:
            self.setup_test_cfg(yml_dict)

    def setup_test_cfg(self, yml_dict):
        self.seq_len = yml_dict.get('test_seq_len', self.seq_len)
        self.data_mask_methods = yml_dict.get('test_data_mask_methods', self.data_mask_methods)

            