import os, sys
sys.path.append(os.path.join(os.getcwd()))
import os.path as osp
import glob
import argparse
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from motion_infiller.data.amass_dataset import AMASSDataset
from motion_infiller.models import model_dict
from motion_infiller.utils.config import Config
from lib.utils.log_utils import TextLogger
from motion_infiller.vis.vis_smpl import SMPLVisualizer
from lib.utils.tools import worker_init_fn, find_last_version, get_checkpoint_path
from lib.utils.vis import hstack_videos
from lib.utils.torch_utils import tensor_to


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='motion_infiller_demo')
parser.add_argument('--split', default='test')
parser.add_argument('--vis_mode', default='smpl')
parser.add_argument('--num_drop_fr', type=int, default=None)
parser.add_argument('--num_seq', type=int, default=2)
parser.add_argument('--num_motion_samp', type=int, default=3)
parser.add_argument('--multi_step', action='store_true', default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--start_ind', type=int, default=0)
parser.add_argument('--version', type=int, default=None)
parser.add_argument('--cp', default='best')
args = parser.parse_args()

cfg = Config(args.cfg, training=False)
seed_everything(args.seed, workers=False)
device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else torch.device('cpu')
torch.torch.set_grad_enabled(False)

# overwrite config for test need
if args.num_drop_fr is not None:
    cfg.data_mask_methods['drop_frames'] = {'min_drop_len': args.num_drop_fr, 'max_drop_len': args.num_drop_fr}


# val datasets
test_dataset = AMASSDataset(cfg.amass_dir, args.split, cfg, training=False, seq_len=cfg.seq_len, ntime_per_epoch=int(1e7))
test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

# checkpoint
version = find_last_version(cfg.cfg_dir) if args.version is None else args.version
checkpoint_dir = f'{cfg.cfg_dir}/version_{version}/checkpoints'
model_cp, cp_name = get_checkpoint_path(checkpoint_dir, args.cp, return_name=True)
print(f'loading check point {model_cp}')

# model
mfiller = model_dict[cfg.model_name].load_from_checkpoint(model_cp, cfg=cfg, strict=False)
mfiller.to(device)
mfiller.eval()

visualizer = SMPLVisualizer(generator_func=None, distance=7, device=device, verbose=False,
                            sample_visible_alltime=False, show_smpl=args.vis_mode in {'smpl', 'all'}, show_skeleton=args.vis_mode in {'jpos', 'all'})


for i, batch in enumerate(test_dataloader):
    if i >= args.num_seq:
        break
    if i < args.start_ind:
        continue
    num_drop_fr = batch["num_drop_fr"][0]
    print(f'{i}/{args.num_seq} seq_name: {batch["seq_name"][0]}, fr_start: {batch["fr_start"][0]} num_mask_fr: {num_drop_fr}')
    
    batch = tensor_to(batch, device)
    output = mfiller.inference(batch, sample_num=args.num_motion_samp, recon=True, multi_step=args.multi_step)

    vid_name = f'out/vis_motion_infiller/{cfg.id}/v{version}_{cp_name}/{args.split}/sd{args.seed}_s{i}_d{num_drop_fr}_%s.mp4'
    # save GT
    visualizer.save_animation_as_video(
        vid_name % 'gt', init_args={'smpl_seq': output, 'mode': 'gt'}, window_size=(1500, 1500), cleanup=True
    )

    if mfiller.has_recon:
        # save recon
        visualizer.save_animation_as_video(
            vid_name % 'recon', init_args={'smpl_seq': output, 'mode': 'recon'}, window_size=(1500, 1500), cleanup=True
        )
        hstack_videos(vid_name % 'recon', vid_name % 'gt', vid_name % 'sbs', verbose=False, text1='Recon', text2='GT', text_color='black')
        os.remove(vid_name % 'recon')

    if mfiller.stochastic:
        # save sample
        visualizer.save_animation_as_video(
            vid_name % 'sample', init_args={'smpl_seq': output, 'mode': 'sample'}, window_size=(1500, 1500), cleanup=True
        )
        hstack_videos(vid_name % 'sample', vid_name % 'gt', vid_name % 'sample_sbs', verbose=False, text1='Sample', text2='GT', text_color='black')
        os.remove(vid_name % 'sample')

    os.remove(vid_name % 'gt')
    
    
    




