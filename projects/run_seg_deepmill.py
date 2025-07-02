# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import math
import argparse
import numpy as np
import  pdb
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='unet_d5')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--depth', type=int, default=5)
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--mode', type=str, default='randinit')
parser.add_argument('--ckpt', type=str, default='\'\'')
parser.add_argument('--ratios', type=float, default=[1], nargs='*')

args = parser.parse_args()
alias = args.alias
gpu = args.gpu
mode = args.mode
ratios = args.ratios
# ratios = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]

module = 'segmentation.py'
script = 'python %s --config configs/seg_deepmill.yaml' % module
data = 'data'
logdir = 'logs/seg_deepmill'

categories = ['models']
names = ['models']
seg_num = [2]
train_num = [4471]
test_num = [1118]
max_epoches = [1500]
max_iters = [1500]

for i in range(len(ratios)):
  for k in range(len(categories)):
    ratio, cat = ratios[i], categories[k]

    mul = 2 if ratios[i] < 0.1 else 1  # longer iterations when data < 10%
    max_epoch = int(max_epoches[k] * ratio * mul)
    milestone1, milestone2 = int(0.5 * max_epoch), int(0.25 * max_epoch)
    # test_every_epoch = int(math.ceil(max_epoch * 0.02))
    test_every_epoch = 50
    take = int(math.ceil(train_num[k] * ratio))
    logs = os.path.join(
        logdir, '{}/{}_{}/ratio_{:.2f}'.format(alias, cat, names[k], ratio))

    cmds = [
        script,
        'SOLVER.gpu {},'.format(gpu),
        'SOLVER.logdir {}'.format(logs),
        'SOLVER.max_epoch {}'.format(max_epoch),
        'SOLVER.milestones {},{}'.format(milestone1, milestone2),
        'SOLVER.test_every_epoch {}'.format(test_every_epoch),
        'SOLVER.ckpt {}'.format(args.ckpt),
        'DATA.train.depth {}'.format(args.depth),
        'DATA.train.filelist {}/filelist/{}_train_val.txt'.format(data, cat),
        'DATA.train.take {}'.format(take),
        'DATA.test.depth {}'.format(args.depth),
        'DATA.test.filelist {}/filelist/{}_test.txt'.format(data, cat),
        'MODEL.stages {}'.format(args.depth - 2),
        'MODEL.nout {}'.format(seg_num[k]),
        'MODEL.name {}'.format(args.model),
        'LOSS.num_class {}'.format(seg_num[k])
    ]

    cmd = ' '.join(cmds)
    print('\n', cmd, '\n')
    # os.system(cmd)
    subprocess.run(cmd)

summary = []
summary.append('names, ' + ', '.join(names) + ', C.mIoU, I.mIoU')
summary.append('train_num, ' + ', '.join([str(x) for x in train_num]))
summary.append('test_num, ' + ', '.join([str(x) for x in test_num]))

for i in range(len(ratios)-1, -1, -1):
  ious = [None] * len(categories)
  for j in range(len(categories)):
    filename = '{}/{}/{}_{}/ratio_{:.2f}/log.csv'.format(
        logdir, alias, categories[j], names[j], ratios[i])
    with open(filename, newline='') as fid:
      lines = fid.readlines()
    last_line = lines[-1]

    pos = last_line.find('test/mIoU:')
    ious[j] = float(last_line[pos+11:pos+16])
  CmIoU = np.array(ious).mean()
  ImIoU = np.sum(np.array(ious)*np.array(test_num)) / np.sum(np.array(test_num))

  ious = [str(iou) for iou in ious] + \
         ['{:.3f}'.format(CmIoU), '{:.3f}'.format(ImIoU)]
  summary.append('Ratio:{:.2f}, '.format(ratios[i]) + ', '.join(ious))

with open('{}/{}/summaries.csv'.format(logdir, alias), 'w') as fid:
  summ = '\n'.join(summary)
  fid.write(summ)
  print(summ)
