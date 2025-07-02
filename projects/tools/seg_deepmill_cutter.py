# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import json
import argparse
import wget
import zipfile
import ssl
import numpy as np
import shutil
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, required=False, default='prepare_dataset',
                    help='The command to run.')
parser.add_argument('--sr', type=float, required=False, default=0.8,
                    help='tran and valid data split ration.')
args = parser.parse_args()

# The following line is to deal with the error of "SSL: CERTIFICATE_VERIFY_FAILED"
# when using wget. (Ref: https://stackoverflow.com/questions/35569042/ssl-certificate-verify-failed-with-python3)

ssl._create_default_https_context = ssl._create_unverified_context

abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(abs_path, 'data')
zip_name = 'raw_data'
txt_folder = os.path.join(root_folder, zip_name)
ply_folder = os.path.join(root_folder, 'points')

categories = ['models']
names = ['models']
seg_num = [2]
# dis = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66]


def normalize_points(input_folder, output_folder):
    """。
    Parameters:
        input_folder (str)
        output_folder (str)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, 'r') as f:
                lines = f.readlines()

            data = []
            for line in lines:
                if line.strip() and not line.strip().startswith('#'):
                    data_data = [x for x in line.split()]
                    if "-nan(ind)" in data_data[3:6]:
                        data_data[3:6] = ["0","0","1"]
                    data.append(data_data)

            data = np.array(data)
            data = data.astype(float)
            xyz = data[:, :3]
            rest = data[:, 3:]

            center = np.mean(xyz, axis=0)
            xyz_centered = xyz - center

            max_extent = np.max(np.abs(xyz_centered), axis=0)
            max_scale = np.max(max_extent)

            xyz_normalized = xyz_centered / max_scale * 0.8
            normalized_data = np.hstack((xyz_normalized, rest))
            np.savetxt(output_path, normalized_data, fmt="%.6f")

            print(f"Processed and saved: {filename}")



def txt_to_ply():

  print('-> Convert txt files to ply files')
  for i, c in enumerate(categories):
    src_folder = os.path.join(txt_folder, c)
    des_folder = os.path.join(ply_folder, c)
    if not os.path.exists(des_folder):
      os.makedirs(des_folder)

    filenames = os.listdir(src_folder)
    for filename in filenames:
      filename_txt = os.path.join(src_folder, filename)
      filename_ply = os.path.join(des_folder, filename[:-4] + '.ply')

      raw = np.loadtxt(filename_txt)
      points = raw[:, :3]
      normals = raw[:, 3:6]
      label = raw[:, 6:7]   # !!! NOTE: the displacement
      label_2 = raw[:, 7:]

      utils.save_points_to_ply(
          filename_ply, points, normals, labels=label, labels_2=label_2, text=False)
      print('Save: ' + os.path.basename(filename_ply))


def generate_filelist():

  print('-> Generate filelists')
  list_folder = os.path.join(root_folder, 'filelist')
  if not os.path.exists(list_folder):
    os.makedirs(list_folder)
  ratio = args.sr

  for i, c in enumerate(categories):

    all_file = os.listdir(os.path.join(txt_folder, c))
    train_val_filelist = []
    test_filelist = []

    for filename in all_file:
      ply_filename = os.path.join(c, filename[:-4] + '.ply')

      cutter_filename = os.path.join(txt_folder, c+'_cutter', filename[:-4] + '_cutter.txt')

      four_numbers = []
      if os.path.exists(cutter_filename):
        cutter_data = np.loadtxt(cutter_filename)
        four_numbers = cutter_data[:4]

      file_entry = '%s %d %.6f %.6f %.6f %.6f' % (ply_filename, i, *four_numbers)

      if len(train_val_filelist) < int(ratio * len(all_file)):
        train_val_filelist.append(file_entry)
      else:
        test_filelist.append(file_entry)

    filelist_name = os.path.join(list_folder, c + '_train_val.txt')
    with open(filelist_name, 'w') as fid:
      fid.write('\n'.join(train_val_filelist))

    filelist_name = os.path.join(list_folder, c + '_test.txt')
    with open(filelist_name, 'w') as fid:
      fid.write('\n'.join(test_filelist))


def prepare_dataset():
    for i, c in enumerate(categories):
        input_folder = os.path.join(txt_folder, c)
        normalize_points(input_folder, input_folder)
    txt_to_ply()
    generate_filelist()


if __name__ == '__main__':
    eval('%s()' % args.run)
