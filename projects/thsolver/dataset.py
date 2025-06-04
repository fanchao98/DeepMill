# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm


def read_file(filename):
  points = np.fromfile(filename, dtype=np.uint8)
  return torch.from_numpy(points)   # convert it to torch.tensor


class Dataset(torch.utils.data.Dataset):

  def __init__(self, root, filelist, transform, read_file=read_file,
               in_memory=False, take: int = -1):
    super(Dataset, self).__init__()
    self.root = root
    self.filelist = filelist
    self.transform = transform
    self.in_memory = in_memory
    self.read_file = read_file
    self.take = take

    self.filenames, self.labels, self.tool_params= self.load_filenames()
    if self.in_memory:
      print('Load files into memory from ' + self.filelist)
      self.samples = [self.read_file(os.path.join(self.root, f))
                      for f in tqdm(self.filenames, ncols=80, leave=False)]

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    sample = (self.samples[idx] if self.in_memory else
              self.read_file(os.path.join(self.root, self.filenames[idx])))
    output = self.transform(sample, idx)  # data augmentation + build octree
    output['label'] = self.labels[idx]
    output['filename'] = self.filenames[idx]
    # 这里确保刀具参数添加到输出中
    output['tool_params'] = self.tool_params[idx]  # 假设在加载数据时已经填充
    return output

  def load_filenames(self):
    filenames, labels, tool_params = [], [], []
    with open(self.filelist) as fid:
      lines = fid.readlines()
    for line in lines:
      tokens = line.split()
      filename = tokens[0].replace('\\', '/')

      # 获取tool_params中的后4位
      if len(tokens) >= 2:
        label = tokens[1]
        # 读取tool_params中的后4位并进行处理，假设tool_params是4维向量
        tool_param = tokens[-4:]  # 获取最后4位
      else:
        label = 0

      filenames.append(filename)
      labels.append(int(label))
      tool_params.append(tool_param)

    num = len(filenames)
    if self.take > num or self.take < 1:
      self.take = num

    return filenames[:self.take], labels[:self.take], tool_params[:self.take]

