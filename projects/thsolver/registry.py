# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

_model_entrypoints = {}
_dataset_entrypoints = {}


def register_model(fn):
  name = fn.__module__.split('.')[-1]
  _model_entrypoints[name] = fn
  return fn


def model_entrypoints(name):
  return _model_entrypoints[name]


def is_model(name):
  return name in _model_entrypoints


def build_model(config, **kwargs):
  name = config.name
  if not is_model(name):
    raise ValueError(f'Unkown model: {name}')
  return model_entrypoints(name)(config, **kwargs)


def register_dataset(fn):
  name = fn.__module__.split('.')[-1]
  _dataset_entrypoints[name] = fn
  return fn


def dataset_entrypoints(name):
  return _dataset_entrypoints[name]


def is_dataset(name):
  return name in _dataset_entrypoints


def build_dataset(config, **kwargs):
  name = config.name
  if not is_dataset(name):
    raise ValueError(f'Unkown dataset: {name}')
  return dataset_entrypoints(name)(config, **kwargs)
