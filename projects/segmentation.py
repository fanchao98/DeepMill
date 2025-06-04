# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import ocnn
import numpy as np
from tqdm import tqdm
from thsolver import Solver

from datasets import (get_seg_shapenet_dataset, get_scannet_dataset,
                      get_kitti_dataset)
import pdb
from sklearn.metrics import f1_score
# The following line is to fix `RuntimeError: received 0 items of ancdata`.
# Refer: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')


class SegSolver(Solver):

    def get_model(self, flags):
        if flags.name.lower() == 'segnet':
            model = ocnn.models.SegNet(
                flags.channel, flags.nout, flags.stages, flags.interp, flags.nempty)
        elif flags.name.lower() == 'unet':
            model = ocnn.models.UNet(
                flags.channel, flags.nout, flags.interp, flags.nempty)
        else:
            raise ValueError
        return model

    def get_dataset(self, flags):
        if flags.name.lower() == 'shapenet':
            return get_seg_shapenet_dataset(flags)
        elif flags.name.lower() == 'scannet':
            return get_scannet_dataset(flags)
        elif flags.name.lower() == 'kitti':
            return get_kitti_dataset(flags)
        else:
            raise ValueError

    def get_input_feature(self, octree):
        flags = self.FLAGS.MODEL
        octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
        data = octree_feature(octree)
        return data

    def process_batch(self, batch, flags):
        def points2octree(points):
            octree = ocnn.octree.Octree(flags.depth, flags.full_depth)
            octree.build_octree(points)
            return octree

        if 'octree' in batch:
            batch['octree'] = batch['octree'].cuda(non_blocking=True)
            batch['points'] = batch['points'].cuda(non_blocking=True)
            # tool_params = batch['tool_params'].cuda(non_blocking=True)
            # batch['tool_params'] = tool_params
        else:
            points = [pts.cuda(non_blocking=True) for pts in batch['points']]
            octrees = [points2octree(pts) for pts in points]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            batch['points'] = ocnn.octree.merge_points(points)
            batch['octree'] = octree
            # tool_params = batch['tool_params'].cuda(non_blocking=True)
            # batch['tool_params'] = tool_params
        return batch


    def model_forward(self, batch):
        octree, points = batch['octree'], batch['points']
        data = self.get_input_feature(octree)
        query_pts = torch.cat([points.points, points.batch_id], dim=1)

        # 从 batch 中提取刀具参数
        tool_params = batch['tool_params']  # 获取刀具参数
        # print(f"Original tool_params: {tool_params}, type: {type(tool_params)}")
        tool_params = [[float(item) for item in row] for row in tool_params]
        tool_params = torch.tensor(tool_params, dtype=torch.float32).cuda() #FC: 需要标注GPU序号
        # print(f"Processed tool_params: {tool_params}, type: {type(tool_params)}, shape: {tool_params.shape}")

        # 将刀具参数传递给模型
        logit_1,logit_2 = self.model.forward(data, octree, octree.depth, query_pts, tool_params)  # 传递刀具参数
        labels = points.labels.squeeze(1)
        label_mask = labels > self.FLAGS.LOSS.mask  # filter labels
        labels_2 = points.labels_2.squeeze(1)
        return logit_1[label_mask], logit_2[label_mask], labels[label_mask], labels_2[label_mask]


    def visualization(self, points, logit, labels,  red_folder,gt_folder):
        # 打开文件进行写入
        with open(red_folder, 'w') as obj_file:
            # 遍历logit张量的每一行
            for i in range(logit.size(0)):  # 遍历每个batch的logit
                # 如果logit第i行的第一个值大于第二个值，则处理对应的点
                if logit[i, 0] > logit[i, 1]:
                    # 获取第i个batch的points
                    batch_points = points[i]

                    # 遍历该batch中的每个点
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")

        with open(gt_folder, 'w') as obj_file:
            # 遍历labels张量的每一行
            for i in range(labels.size(0)):  # 遍历每个batch的labels
                # 如果labels第i行的值为0，则处理对应的点
                if labels[i] == 0:
                    batch_points = points[i]  # 获取第i个batch的points
                    # 遍历该batch中的每个点并写入到.obj文件
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")
                
    def visualization1(self, points, logit, labels,  red_folder,gt_folder):
        # 打开文件进行写入
        with open(red_folder, 'w') as obj_file:
            # 遍历logit张量的每一行
            for i in range(logit.size(0)):  # 遍历每个batch的logit
                # 如果logit第i行的第一个值大于第二个值，则处理对应的点
                if logit[i, 0] < logit[i, 1]:
                    # 获取第i个batch的points
                    batch_points = points[i]

                    # 遍历该batch中的每个点
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")

        with open(gt_folder, 'w') as obj_file:
            # 遍历labels张量的每一行
            for i in range(labels.size(0)):  # 遍历每个batch的labels
                # 如果labels第i行的值为0，则处理对应的点
                if labels[i] == 1:
                    batch_points = points[i]  # 获取第i个batch的points
                    # 遍历该batch中的每个点并写入到.obj文件
                    obj_file.write(f"v {batch_points.points[0]} {batch_points.points[1]} {batch_points.points[2]}\n")


    def train_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.train)
        logit_1,logit_2, label, label_2 = self.model_forward(batch)
        loss_1 = self.loss_function(logit_1, label)
        loss_2 = self.loss_function(logit_2, label_2)
        loss = (loss_1 + loss_2)/2
        accu_1 = self.accuracy(logit_1, label)
        accu_2 = self.accuracy(logit_2, label_2)
        accu = (accu_1 + accu_2)/2

        pred_1 = logit_1.argmax(dim=-1)  # 假设 logit_1 是 logits 形式，需要用 argmax 选取预测类别
        pred_2 = logit_2.argmax(dim=-1)
        # 这里使用 f1_score 函数，假设 label 和 label_2 都是 0 和 1 的整数标签
        f1_score_1 = f1_score(label.cpu().numpy(), pred_1.cpu().numpy(), average='binary')
        f1_score_2 = f1_score(label_2.cpu().numpy(), pred_2.cpu().numpy(), average='binary')
        f1_score_avg = (f1_score_1 + f1_score_2) / 2

        return {'train/loss': loss, 'train/accu': accu, 'train/accu_red': accu_1, 'train/accu_green': accu_2,
                'train/f1_red': torch.tensor(f1_score_1, dtype=torch.float32).cuda(), 'train/f1_green': torch.tensor(f1_score_2, dtype=torch.float32).cuda(), 'train/f1_avg': torch.tensor(f1_score_avg, dtype=torch.float32).cuda()}
        # return {'train/loss': loss, 'train/accu': accu,'train/accu_red': accu_1,'train/accu_green': accu_2,
        # 'train/f1_red': f1_score_1,'train/f1_green': f1_score_2,'train/f1_avg': f1_score_avg}



    def test_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        with torch.no_grad():
            logit_1,logit_2, label, label_2 = self.model_forward(batch)
        # self.visualization(batch['points'], logit, label, ".\\data\\vis\\"+batch['filename'][0][:-4]+".obj") #FC:目前可视化只支持test的batch size=1
        loss_1 = self.loss_function(logit_1, label)
        loss_2 = self.loss_function(logit_2, label_2)
        loss = (loss_1 + loss_2) / 2
        accu_1 = self.accuracy(logit_1, label)
        accu_2 = self.accuracy(logit_2, label_2)
        accu = (accu_1 + accu_2) / 2
        num_class = self.FLAGS.LOSS.num_class
        IoU, insc, union = self.IoU_per_shape(logit_1, label, num_class)

        folders = [
            './visual/red_points',
            './visual/GT_red',
            './visual/green_points',
            './visual/GT_green'
        ]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)
              
        red_folder = os.path.join(r"./visual/red_points",
                                  batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
                                      0] + ".obj")
        gt_red_folder = os.path.join(r"./visual/GT_red",
                                     batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
                                         0] + ".obj")
        green_folder = os.path.join(r'./visual/green_points',
                                    batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
                                        0] + ".obj")
        gt_green_folder = os.path.join(r'./visual/GT_green',
                                       batch['filename'][0].split("/")[-1].split(".")[0].split("_collision_detection")[
                                           0] + ".obj")
        self.visualization(batch['points'], logit_1, label, red_folder, gt_red_folder)
        self.visualization1(batch['points'], logit_2, label_2, green_folder, gt_green_folder)
        pred_1 = logit_1.argmax(dim=-1)
        pred_2 = logit_2.argmax(dim=-1)
        # 这里使用 f1_score 函数，假设 label 和 label_2 都是 0 和 1 的整数标签
        f1_score_1 = f1_score(label.cpu().numpy(), pred_1.cpu().numpy(), average='binary')
        f1_score_2 = f1_score(label_2.cpu().numpy(), pred_2.cpu().numpy(), average='binary')
        f1_score_avg = (f1_score_1 + f1_score_2) / 2

        names = ['test/loss', 'test/accu', 'test/accu_red','test/accu_green','test/mIoU', 'test/f1_red','test/f1_green','test/f1_avg'] + \
                ['test/intsc_%d' % i for i in range(num_class)] + \
                ['test/union_%d' % i for i in range(num_class)]
        tensors = [loss, accu, accu_1, accu_2, IoU, torch.tensor(f1_score_1, dtype=torch.float32).cuda(),
                   torch.tensor(f1_score_2, dtype=torch.float32).cuda(),
                   torch.tensor(f1_score_avg, dtype=torch.float32).cuda()] + insc + union
        return dict(zip(names, tensors))


    def eval_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        with torch.no_grad():
            logit, _ = self.model_forward(batch)
        prob = torch.nn.functional.softmax(logit, dim=1)

        # split predictions
        inbox_masks = batch['inbox_mask']
        npts = batch['points'].batch_npt.tolist()
        probs = torch.split(prob, npts)

        # merge predictions
        batch_size = len(inbox_masks)
        for i in range(batch_size):
            # The point cloud may be clipped when doing data augmentation. The
            # `inbox_mask` indicates which points are clipped. The `prob_all_pts`
            # contains the prediction for all points.
            prob = probs[i].cpu()
            inbox_mask = inbox_masks[i].to(prob.device)
            prob_all_pts = prob.new_zeros([inbox_mask.shape[0], prob.shape[1]])
            prob_all_pts[inbox_mask] = prob

            # Aggregate predictions across different epochs
            filename = batch['filename'][i]
            self.eval_rst[filename] = self.eval_rst.get(filename, 0) + prob_all_pts

            # Save the prediction results in the last epoch
            if self.FLAGS.SOLVER.eval_epoch - 1 == batch['epoch']:
                full_filename = os.path.join(self.logdir, filename[:-4] + '.eval.npz')
                curr_folder = os.path.dirname(full_filename)
                if not os.path.exists(curr_folder): os.makedirs(curr_folder)
                np.savez(full_filename, prob=self.eval_rst[filename].cpu().numpy())

    def result_callback(self, avg_tracker, epoch):
        r''' Calculate the part mIoU for PartNet and ScanNet.
        '''

        iou_part = 0.0
        avg = avg_tracker.average()

        # Labels smaller than `mask` is ignored. The points with the label 0 in
        # PartNet are background points, i.e., unlabeled points
        mask = self.FLAGS.LOSS.mask + 1
        num_class = self.FLAGS.LOSS.num_class
        for i in range(mask, num_class):
            instc_i = avg['test/intsc_%d' % i]
            union_i = avg['test/union_%d' % i]
            iou_part += instc_i / (union_i + 1.0e-10)
        iou_part = iou_part / (num_class - mask)

        avg_tracker.update({'test/mIoU_part': torch.Tensor([iou_part])})
        tqdm.write('=> Epoch: %d, test/mIoU_part: %f' % (epoch, iou_part))

    def loss_function(self, logit, label):
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(logit, label.long())
        return loss

    def accuracy(self, logit, label):
        pred = logit.argmax(dim=1)
        accu = pred.eq(label).float().mean()
        return accu

    def IoU_per_shape(self, logit, label, class_num):
        pred = logit.argmax(dim=1)

        IoU, valid_part_num, esp = 0.0, 0.0, 1.0e-10
        intsc, union = [None] * class_num, [None] * class_num
        for k in range(class_num):
            pk, lk = pred.eq(k), label.eq(k)
            intsc[k] = torch.sum(torch.logical_and(pk, lk).float())
            union[k] = torch.sum(torch.logical_or(pk, lk).float())

            valid = torch.sum(lk.any()) > 0
            valid_part_num += valid.item()
            IoU += valid * intsc[k] / (union[k] + esp)

        # Calculate the shape IoU for ShapeNet
        IoU /= valid_part_num + esp
        return IoU, intsc, union


if __name__ == "__main__":

    SegSolver.main()
