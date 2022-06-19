import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
import pickle
import numpy as np
import glob 


class AMASSDataset(Dataset):

    def __init__(self, dataset_dir, split, cfg=None, training=True, seq_len=64, ntime_per_epoch=10000):
        self.cfg = cfg
        self.data = pickle.load(open(f'{dataset_dir}/amass_{split}.pkl', 'rb'))
        self.data_jpos = pickle.load(open(f'{dataset_dir}/amass_{split}_jpos.pkl', 'rb'))
        self.sequences = list(self.data.keys())
        self.split = split
        self.training = training
        self.seq_len = seq_len
        self.ntime_per_epoch = ntime_per_epoch
        self.epoch_init_seed = None
        # compute sampling probablity
        self.seq_lengths = np.array([x.shape[0] for x in self.data.values()])
        if cfg is not None and cfg.seq_sampling_method == 'length':
            self.seq_prob = self.seq_lengths / self.seq_lengths.sum()
        else:
            self.seq_prob = None

    def __len__(self):
       return self.ntime_per_epoch // self.seq_len

    def set_seq_len(self, seq_len):
        self.seq_len = seq_len        

    def random_sample(self, idx=0):
        if self.epoch_init_seed is None:
            # the above step is necessary for lightning's ddp parallel computing because each node gets a subset (idx) of the dataset
            self.epoch_init_seed = (np.random.get_state()[1][0] * len(self) + idx) % int(1e8)
            np.random.seed(self.epoch_init_seed)
            # print('epoch_init_seed', self.epoch_init_seed)
        
        sind = np.random.choice(len(self.sequences), p=self.seq_prob)
        seq = self.sequences[sind]
        seq_jpos, seq_jpos_noshape = self.data_jpos[seq]

        if self.seq_len <= self.data[seq].shape[0]:
            fr_start = np.random.randint(self.data[seq].shape[0] - self.seq_len + 1)
            seq_data = self.data[seq][fr_start: fr_start + self.seq_len].astype(np.float32)
            frame_loss_mask = np.ones((self.seq_len, 1)).astype(np.float32)
            eff_seq_len = self.seq_len   # effective seq
            # joint pos
            jpos = seq_jpos[fr_start: fr_start + self.seq_len].astype(np.float32)
            jpos_noshape = seq_jpos_noshape[fr_start: fr_start + self.seq_len].astype(np.float32)
        else:
            fr_start = 0
            seq_data = np.vstack([self.data[seq], np.tile(self.data[seq][[-1]], (self.seq_len - self.data[seq].shape[0], 1))]).astype(np.float32)
            frame_loss_mask = np.zeros((self.seq_len, 1)).astype(np.float32)
            frame_loss_mask[:self.data[seq].shape[0]] = 1.0
            eff_seq_len = self.data[seq].shape[0]   # effective seq
            # joint pos
            jpos = np.vstack([seq_jpos, np.tile(seq_jpos[[-1]], (self.seq_len - seq_jpos.shape[0], 1, 1))]).astype(np.float32)
            jpos_noshape = np.vstack([seq_jpos_noshape, np.tile(seq_jpos_noshape[[-1]], (self.seq_len - seq_jpos_noshape.shape[0], 1, 1))]).astype(np.float32)

        data = {
            'trans': seq_data[:, :3],
            'pose': seq_data[:, 3:75],
            'shape': seq_data[:, 75:],
            'seq_name': seq,
            'frame_loss_mask': frame_loss_mask,
            'fr_start': fr_start,
            'eff_seq_len': eff_seq_len,
            'joint_pos_shape': jpos[:, 1:, :].reshape(jpos.shape[0], -1),
            'joint_pos_noshape': jpos_noshape[:, 1:, :].reshape(jpos_noshape.shape[0], -1)
            # 'seq_ind': sind,
            # 'idx': idx,
        }

        # mask
        self.generate_mask(data)

        # gaussian smoothing for data augmentation
        if self.cfg is not None and self.cfg.pose_gaussian_smooth is not None:
            in_body_pose = seq_data[:, 6:75]
            d = self.cfg.pose_gaussian_smooth
            if np.random.binomial(1, d['prob']):
                sigma = np.random.uniform(d['sigma_lb'], d['sigma_ub'])
                # print('smoothing', sigma)
                in_body_pose = gaussian_filter1d(in_body_pose.copy(), sigma=sigma, axis=0, mode='nearest')
            in_body_pose *= data['pose_mask'][:, 3:]
            data['in_body_pose'] = in_body_pose
        return data

    def generate_mask(self, data):
        mask_methods = self.cfg.data_mask_methods if self.cfg is not None else dict()
        pose_mask = np.ones_like(data['pose'])
        frame_mask = np.ones(data['pose'].shape[0]).astype(np.float32)
        for method, specs in mask_methods.items():
            if method == 'drop_frames':
                preserve_first_n = specs.get('preserve_first_n', 1)
                preserve_last_n = specs.get('preserve_last_n', 0)
                drop_len = np.random.randint(specs['min_drop_len'], specs['max_drop_len'] + 1)
                start_fr_min = preserve_first_n
                start_fr_max = min(data['pose'].shape[0] - drop_len + 1 - preserve_last_n, data['eff_seq_len'])
                start_fr = np.random.randint(start_fr_min, start_fr_max)
                end_fr = min(start_fr + drop_len, pose_mask.shape[0])
                pose_mask[start_fr: end_fr] = 0.0
                frame_mask[start_fr: end_fr] = 0.0
                data['num_drop_fr'] = end_fr - start_fr
        data['pose_mask'] = pose_mask
        data['frame_mask'] = frame_mask

    def __getitem__(self, idx):
        return self.random_sample(idx)


if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)
    amass_dir = 'datasets/amass_processed/v1'

    dataset = AMASSDataset(amass_dir, 'test', seq_len=200)
    print(f'dataset has {len(dataset)} data')

    batch_size = 5
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    batch_rand = dataset.random_sample()
    print(batch['pose'].shape)