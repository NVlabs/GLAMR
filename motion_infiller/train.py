import os, sys
sys.path.append(os.path.join(os.getcwd()))
import os.path as osp
import argparse
import tempfile
import subprocess
import glob
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from motion_infiller.data.amass_dataset import AMASSDataset
from motion_infiller.models import model_dict
from motion_infiller.utils.config import Config
from lib.utils.log_utils import TextLogger
from lib.utils.tools import worker_init_fn, find_last_version, get_checkpoint_path


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None)
parser.add_argument('--tmp', action='store_true', default=False)
parser.add_argument('--ngpus', type=int, default=1)
parser.add_argument('--gpu_ids', default=None)
parser.add_argument('--nworkers', type=int, default=8)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--max_epochs', type=int, default=None)
parser.add_argument('--save_n_epochs', type=int, default=None)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--version', type=int, default=None)
parser.add_argument('--cp', default='last')
parser.add_argument('--profiler', default=None)
parser.add_argument('--vis_num_seq', type=int, default=0)
parser.add_argument('--vis_ndrop', type=int, default=None)
args = parser.parse_args()


process_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ.keys() else -1
gpu_rank = max(process_rank, 0)
cfg = Config(args.cfg, tmp=args.tmp, training=True)
seed_everything(cfg.seed, workers=False)
if args.ngpus > 1:
    cfg.batch_size //= args.ngpus
    # cfg.lr *= args.ngpus      # only necessary when loss is proportional to batch size
gpu_ids = None if args.gpu_ids is None else [int(x) for x in args.gpu_ids.split(',')]

# additional setup for debugging and speed optimization
if args.debug:
    torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)

# train datasets
train_dataset = AMASSDataset(cfg.amass_dir, 'train', cfg, seq_len=cfg.seq_len, ntime_per_epoch=cfg.train_ntime_per_epoch)
train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=args.nworkers, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
# val datasets
val_dataset = AMASSDataset(cfg.amass_dir, 'test', cfg, seq_len=cfg.seq_len, ntime_per_epoch=cfg.val_ntime_per_epoch)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=args.nworkers, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
# model
mfiller = model_dict[cfg.model_name](cfg)


# logger
if args.resume:
    version = find_last_version(cfg.cfg_dir) if args.version is None else args.version
else:
    version = None
max_epochs = cfg.max_epochs if args.max_epochs is None else args.max_epochs


if process_rank == -1:
    # only the main process does logging
    tb_logger = TensorBoardLogger(f'{cfg.cfg_dir}', version=version, name='')
    version = tb_logger.version
    text_logger = TextLogger(f'{tb_logger.log_dir}/logs/log.txt', cfg=cfg, max_epochs=max_epochs)
    checkpoint_dir = f'{tb_logger.log_dir}/checkpoints'
    loggers = [tb_logger, text_logger]
    if not args.tmp:
        wandb_logger = WandbLogger(project="GLAMR.motion_infiller", name=f'{cfg.id}_v{version}', save_dir=f'{tb_logger.log_dir}')
        loggers.append(wandb_logger)
else:
    # child processes
    if args.resume:
        checkpoint_dir = f'{cfg.cfg_dir}/version_{version}/checkpoints' # used only when resuming training
    else:
        checkpoint_dir = tempfile.mkdtemp()
    loggers = None
print(f'process: {process_rank}, checkpoint_dir: {checkpoint_dir}')

checkpoint_epoch_cb = ModelCheckpoint(
    monitor='val_loss',
    dirpath=checkpoint_dir,
    filename='model-{epoch:04d}',
    save_last=False,
    save_top_k=-1,
    mode='min',
    every_n_val_epochs=cfg.save_n_epochs if args.save_n_epochs is None else args.save_n_epochs
)
checkpoint_best_cb = ModelCheckpoint(
    monitor='val_loss',
    dirpath=checkpoint_dir,
    filename='model-best-{epoch:04d}',
    save_last=True,
    save_top_k=1,
    mode='min'
)
callbacks = [checkpoint_epoch_cb, checkpoint_best_cb]

resume_cp = get_checkpoint_path(checkpoint_dir, args.cp) if args.resume else None

# trainer
trainer = pl.Trainer(
    logger=loggers,
    callbacks=callbacks,
    gpus=gpu_ids if gpu_ids is not None else args.ngpus,
    auto_select_gpus=False,
    accelerator='ddp' if args.ngpus > 1 else None,
    precision=args.precision,
    resume_from_checkpoint=resume_cp,
    progress_bar_refresh_rate=0, 
    max_epochs=max_epochs,
    profiler=args.profiler,
    gradient_clip_val=cfg.gradient_clip_val
)
trainer.fit(mfiller, train_dataloader, val_dataloader)


# save visualization videos
if process_rank == -1 and args.vis_num_seq > 0:
    cmd = f'python motion_infiller/vis_res.py --cfg {cfg.id} --num_seq {args.vis_num_seq}'
    if args.vis_ndrop is not None:
        cmd += f' --num_drop_fr {args.vis_ndrop}'
    subprocess.run(cmd.split(' '))
