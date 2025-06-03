import torch
import torch.nn
from typing import Dict
import ocnn
from ocnn.octree import Octree
import math

class UNet(torch.nn.Module):
    r''' Octree-based UNet for segmentation.
    '''

    def __init__(self, in_channels: int, out_channels: int, interp: str = 'linear',
                 nempty: bool = False, **kwargs):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nempty = nempty
        self.config_network()
        self.encoder_stages = len(self.encoder_blocks)
        self.decoder_stages = len(self.decoder_blocks)
        self.batch_size = 1

        # encoder
        self.conv1 = ocnn.modules.OctreeConvBnRelu(
            in_channels, self.encoder_channel[0], nempty=nempty)
        self.downsample = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
            self.encoder_channel[i], self.encoder_channel[i+1], kernel_size=[2],
            stride=2, nempty=nempty) for i in range(self.encoder_stages)])
        self.encoder = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            self.encoder_channel[i+1], self.encoder_channel[i + 1],
            self.encoder_blocks[i], self.bottleneck, nempty, self.resblk)
            for i in range(self.encoder_stages)])

        # decoder
        channel = [self.decoder_channel[i+1] + self.encoder_channel[-i-2]
                   for i in range(self.decoder_stages)]
        channel[3] =  channel[3] + 256
        channel[2] = channel[2] + 256
        channel[1] = channel[1] + 256
        channel[0] = channel[0] + 256
        self.upsample = torch.nn.ModuleList([ocnn.modules.OctreeDeconvBnRelu(
            self.decoder_channel[i], self.decoder_channel[i+1], kernel_size=[2],
            stride=2, nempty=nempty) for i in range(self.decoder_stages)])
        self.decoder = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            channel[i], self.decoder_channel[i+1],
            self.decoder_blocks[i], self.bottleneck, nempty, self.resblk)
            for i in range(self.decoder_stages)])

        # header
        self.octree_interp = ocnn.nn.OctreeInterp(interp, nempty)
        self.header = torch.nn.Sequential(
            ocnn.modules.Conv1x1BnRelu(self.decoder_channel[-1], self.head_channel),
            ocnn.modules.Conv1x1(self.head_channel, self.out_channels, use_bias=True))
        self.header_2 = torch.nn.Sequential(
            ocnn.modules.Conv1x1BnRelu(self.decoder_channel[-1], self.head_channel),
            ocnn.modules.Conv1x1(self.head_channel, self.out_channels, use_bias=True))


        self.fc_module_1 = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
        )

        self.fc_module_2 = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
        )
        self.fc_module_3 = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
        )
        self.fc_module_4 = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
        )

    def config_network(self):
        r''' Configure the network channels and Resblock numbers.
        '''
        self.encoder_channel = [32, 32, 64, 128, 256]
        self.decoder_channel = [256, 256, 128, 96, 96]
        self.encoder_blocks = [2, 3, 4, 6]
        self.decoder_blocks = [2, 2, 2, 2]
        self.head_channel = 64
        self.bottleneck = 1
        self.resblk = ocnn.modules.OctreeResBlock2

    def unet_encoder(self, data: torch.Tensor, octree: Octree, depth: int):
        r''' The encoder of the U-Net.
        '''
        convd = dict()
        convd[depth] = self.conv1(data, octree, depth)
        for i in range(self.encoder_stages):
            d = depth - i
            conv = self.downsample[i](convd[d], octree, d)
            convd[d-1] = self.encoder[i](conv, octree, d-1)
        return convd

    def unet_decoder(self, convd: Dict[int, torch.Tensor], octree: Octree, depth: int,tool_features_1,tool_features_2,tool_features_3,tool_features_4):
        r''' The decoder of the U-Net.
        '''
        deconv = convd[depth]
        for i in range(self.decoder_stages):
            d = depth + i
            deconv = self.upsample[i](deconv, octree, d)

            copy_counts = octree.batch_nnum[i+2]
            expanded_tool_features = []
            if i == 0:
                for j in range(tool_features_1.size(0)):
                    expanded_tool_features.append(tool_features_1[j, :].repeat(copy_counts[j], 1))
            if i == 1:
                for j in range(tool_features_2.size(0)):
                    expanded_tool_features.append(tool_features_2[j, :].repeat(copy_counts[j], 1))
            if i == 2:
                for j in range(tool_features_3.size(0)):
                    expanded_tool_features.append(tool_features_3[j, :].repeat(copy_counts[j], 1))
            if i == 3:
                for j in range(tool_features_4.size(0)):
                    expanded_tool_features.append(tool_features_4[j, :].repeat(copy_counts[j], 1))
            expanded_tool_features = torch.cat(expanded_tool_features, dim=0)
            # tool_features = tool_features.repeat(math.ceil(deconv.size(0) / tool_features.size(0)), 1)
            deconv = torch.cat([expanded_tool_features, deconv], dim=1)  # skip connections

            deconv = torch.cat([convd[d+1], deconv], dim=1)  # skip connections
            deconv = self.decoder[i](deconv, octree, d+1)
        return deconv

    def forward(self, data: torch.Tensor, octree: Octree, depth: int,
                query_pts: torch.Tensor, tool_params: torch.Tensor):
        r''' Forward pass with tool parameters incorporated.
        '''

        convd = self.unet_encoder(data, octree, depth)

        tool_features_1 = self.fc_module_1(tool_params)
        tool_features_2 = self.fc_module_2(tool_params)
        tool_features_3 = self.fc_module_3(tool_params)
        tool_features_4 = self.fc_module_4(tool_params)

        deconv = self.unet_decoder(convd, octree, depth - self.encoder_stages,tool_features_1,tool_features_2,tool_features_3,tool_features_4)

        interp_depth = depth - self.encoder_stages + self.decoder_stages
        # print(f"deconv shape: {deconv.shape}")
        # print(f"octree depth: {interp_depth}, query_pts shape: {query_pts.shape}")
        feature = self.octree_interp(deconv, octree, interp_depth, query_pts)
        # print(f"query_pts shape: {query_pts.shape}")
        # print(f"deconv batch size: {deconv.shape[0]}")
        # print(f"octree batch size: {octree.batch_size}")
        # print(f"query_pts batch size: {query_pts.shape[0]}")
        logits_1 = self.header(feature)
        logits_2 = self.header_2(feature)
        return logits_1,logits_2