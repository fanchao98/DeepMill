# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

# autopep8: off
import os
import sys
import shutil
import argparse
from datetime import datetime
from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)

_C.BASE = ['']

# SOLVER related parameters
_C.SOLVER = CN(new_allowed=True)
_C.SOLVER.alias             = ''         # The experiment alias
_C.SOLVER.gpu               = (0,)       # The gpu ids
_C.SOLVER.run               = 'train'    # Choose from train or test

_C.SOLVER.logdir            = 'logs'     # Directory where to write event logs
_C.SOLVER.ckpt              = ''         # Restore weights from checkpoint file
_C.SOLVER.ckpt_num          = 10         # The number of checkpoint kept

_C.SOLVER.type              = 'sgd'      # Choose from sgd or adam
_C.SOLVER.weight_decay      = 0.0005     # The weight decay on model weights
_C.SOLVER.clip_grad         = -1.0       # Clip gradient norm (-1: disable)
_C.SOLVER.max_epoch         = 300        # Maximum training epoch
_C.SOLVER.warmup_epoch      = 20         # The warmup epoch number
_C.SOLVER.warmup_init       = 0.001      # The initial ratio of the warmup
_C.SOLVER.eval_epoch        = 1          # Maximum evaluating epoch
_C.SOLVER.eval_step         = -1         # Maximum evaluating steps
_C.SOLVER.test_every_epoch  = 10         # Test model every n training epochs
_C.SOLVER.log_per_iter      = -1         # Output log every k training iteration
_C.SOLVER.best_val          = 'min:loss' # The best validation metric

_C.SOLVER.lr_type           = 'step'     # Learning rate type: step or cos
_C.SOLVER.lr                = 0.1        # Initial learning rate
_C.SOLVER.lr_min            = 0.0001     # The minimum learning rate
_C.SOLVER.gamma             = 0.1        # Learning rate step-wise decay
_C.SOLVER.milestones        = (120,180,) # Learning rate milestones
_C.SOLVER.lr_power          = 0.9        # Used in poly learning rate

# _C.SOLVER.dist_url          = 'tcp://localhost:10001'
_C.SOLVER.port              = 10001      # The port number for distributed training
_C.SOLVER.progress_bar      = True       # Enable the progress_bar or not
_C.SOLVER.rand_seed         = -1         # Fix the random seed if larger than 0
_C.SOLVER.empty_cache       = True       # Empty cuda cache periodically

# DATA related parameters
_C.DATA = CN(new_allowed=True)
_C.DATA.train = CN(new_allowed=True)
_C.DATA.train.name          = ''          # The name of the dataset
_C.DATA.train.disable       = False       # Disable this dataset or not
_C.DATA.train.pin_memory    = True

# For octree building
_C.DATA.train.depth         = 5           # The octree depth
_C.DATA.train.full_depth    = 2           # The full depth
_C.DATA.train.adaptive      = False       # Build the adaptive octree

# For transformation
_C.DATA.train.orient_normal = ''          # Used to re-orient normal directions

# For data augmentation
_C.DATA.train.distort       = False       # Whether to apply data augmentation
_C.DATA.train.scale         = 0.0         # Scale the points
_C.DATA.train.uniform       = False       # Generate uniform scales
_C.DATA.train.jitter        = 0.0         # Jitter the points
_C.DATA.train.interval      = (1, 1, 1)   # Use interval&angle to generate random angle
_C.DATA.train.angle         = (180, 180, 180)
_C.DATA.train.flip          = (0.0, 0.0, 0.0)


# For data loading
_C.DATA.train.location      = ''          # The data location
_C.DATA.train.filelist      = ''          # The data filelist
_C.DATA.train.batch_size    = 32          # Training data batch size
_C.DATA.train.take          = -1          # Number of samples used for training
_C.DATA.train.num_workers   = 4           # Number of workers to load the data
_C.DATA.train.shuffle       = False       # Shuffle the input data
_C.DATA.train.in_memory     = False       # Load the training data into memory


_C.DATA.test = _C.DATA.train.clone()
_C.DATA.test.num_workers    = 2

# MODEL related parameters
_C.MODEL = CN(new_allowed=True)
_C.MODEL.name               = ''          # The name of the model
_C.MODEL.feature            = 'ND'        # The input features
_C.MODEL.channel            = 3           # The input feature channel
_C.MODEL.nempty             = False       # Perform Octree Conv on non-empty octree nodes

_C.MODEL.sync_bn            = False       # Use sync_bn when training the network
_C.MODEL.use_checkpoint     = False       # Use checkpoint to save memory
_C.MODEL.find_unused_parameters = False   # Used in DistributedDataParallel


# loss related parameters
_C.LOSS = CN(new_allowed=True)
_C.LOSS.name                = ''          # The name of the loss
_C.LOSS.num_class           = 40          # The class number for the cross-entropy loss
_C.LOSS.label_smoothing     = 0.0         # The factor of label smoothing


# backup the commands
_C.SYS = CN(new_allowed=True)
_C.SYS.cmds              = ''             # Used to backup the commands

FLAGS = _C


def _load_from_file(filename):
  cfgs = []
  bases = [filename]
  while len(bases) > 0:
    base = bases.pop(0)
    if base:
      with open(base, 'r') as fid:
        cfg = CN.load_cfg(fid)
      cfgs.append(cfg)
      if 'BASE' in cfg:
        assert isinstance(cfg.BASE, list), 'BASE should be a list'
        bases += cfg.BASE
  cfgs.reverse()
  return cfgs


def _update_config(FLAGS, args):
  FLAGS.defrost()
  if args.config:
    # FLAGS.merge_from_file(args.config)
    cfgs = _load_from_file(args.config)
    for cfg in cfgs:
      FLAGS.merge_from_other_cfg(cfg)
  if args.opts:
    FLAGS.merge_from_list(args.opts)
  FLAGS.SYS.cmds = 'python ' + ' '.join(sys.argv)

  # update logdir
  alias = FLAGS.SOLVER.alias.lower()
  if 'time' in alias:  # 'time' is a special keyword
    alias = alias.replace('time', datetime.now().strftime('%m%d%H%M')) #%S
  if alias != '':
    FLAGS.SOLVER.logdir += '_' + alias
  FLAGS.freeze()


def _backup_config(FLAGS, args):
  logdir = FLAGS.SOLVER.logdir
  os.makedirs(logdir, exist_ok=True)

  # copy the file to logdir
  if args.config:
    shutil.copy2(args.config, logdir)

  # dump all configs
  filename = os.path.join(logdir, 'all_configs.yaml')
  with open(filename, 'w') as fid:
    fid.write(FLAGS.dump())


def _set_env_var(FLAGS):
  gpus = ','.join([str(a) for a in FLAGS.SOLVER.gpu])
  os.environ['CUDA_VISIBLE_DEVICES'] = gpus


def get_config():
  return FLAGS

def parse_args(backup=True):
  parser = argparse.ArgumentParser(description='The configs')
  parser.add_argument('--config', type=str,
                      help='experiment configure file name')
  parser.add_argument('opts', nargs=argparse.REMAINDER,
                      help="Modify config options using the command-line")

  args = parser.parse_args()
  _update_config(FLAGS, args)
  if backup:
    _backup_config(FLAGS, args)
  # _set_env_var(FLAGS)
  return FLAGS


if __name__ == '__main__':
  flags = parse_args(backup=False)
  print(flags)
