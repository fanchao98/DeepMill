# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from thsolver import Dataset
from ocnn.octree import Points
from ocnn.dataset import CollateBatch

from .utils import ReadPly, Transform
import os

class ShapeNetTransform(Transform):

  def preprocess(self, sample: dict, idx: int):

    xyz = torch.from_numpy(sample['points']).float()
    normal = torch.from_numpy(sample['normals']).float()
    labels = torch.from_numpy(sample['labels']).float()
    labels_2 = torch.from_numpy(sample['labels_2']).float()
    points = Points(xyz, normal, labels=labels.unsqueeze(1), labels_2=labels_2.unsqueeze(1))

    # !NOTE: Normalize the points into one unit sphere in [-0.8, 0.8]
    # bbmin, bbmax = points.bbox()
    # points.normalize(bbmin, bbmax, scale=0.8)

    return {'points': points}



def load_tool_params(filename, tool_params_dir):
  """ 读取每个样本对应的刀具参数 (四个浮点数) """
  tool_param_file = os.path.join(tool_params_dir, filename.replace('.ply', '.txt'))  # 假设文件名对应
  with open(tool_param_file, 'r') as f:
    tool_params = list(map(float, f.readline().split()[1:]))  # 跳过第一个文件名部分
  return torch.tensor(tool_params, dtype=torch.float32)



def get_seg_shapenet_dataset(flags):
  transform = ShapeNetTransform(flags)
  read_ply = ReadPly(has_normal=True, has_label=True)
  collate_batch = CollateBatch(merge_points=True)

  # 创建数据集
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_ply, take=flags.take )

  # # 遍历数据集并为每个样本加载刀具参数
  # for item in dataset:
  #   filename = item['filename']  # 获取当前样本的文件名
  #   tool_params = load_tool_params(filename, tool_params_dir)  # 加载刀具参数
  #   item['tool_params'] = tool_params  # 将刀具参数添加到数据项中

  return dataset, collate_batch