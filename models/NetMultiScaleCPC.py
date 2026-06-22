# -*- coding: utf-8 -*-
"""
Multi-scale CPC version of SFCNet (Multi-Scale Category Prototype Contrast)
Also add CPC modules at decoder levels F2, F3, F4 to form scale-aware prototype contrastive learning

Novelty:
1. features at different scales need different prototype representations - small targets rely on low-level features, large targets on high-level features
2. multi-scale prototypes form a hierarchical prototype structure
3. support an ablation of shared vs independent prototypes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import (Conv, UpSampler, FAI, DWT, BasicBlockL, BasicBlockH, CIU,
                          CategoryPrototypeContrast, CategoryPrototypeContrastAdaptiveTemp)
from models.smt import smt_t


class NetMultiScaleCPC(nn.Module):
    """
    Multi-scale CPC version of SFCNet
    Add a CPC module at all four decoder levels F1, F2, F3, F4
    """

    def __init__(self, num_classes=4, temperature=0.07, enable_refinement=True,
                 adaptive_temp=False, shared_prototypes=False):
        super(NetMultiScaleCPC, self).__init__()

        self.num_classes = num_classes
        self.temperature = temperature
        self.enable_refinement = enable_refinement
        self.adaptive_temp = adaptive_temp
        self.shared_prototypes = shared_prototypes
        channels = [64, 128, 256, 512]
        self.channels = channels
        self.rgb_swin = smt_t()

        # --------------------------------------------------------------------
        self.conv_rgb0 = nn.Sequential(nn.Conv2d(self.channels[0], self.channels[1], 1, 1, 0), nn.BatchNorm2d(self.channels[1]), nn.ReLU())
        self.conv_rgb1 = nn.Sequential(nn.Conv2d(self.channels[1], self.channels[1], 1, 1, 0), nn.BatchNorm2d(self.channels[1]), nn.ReLU())
        self.conv_rgb2 = nn.Sequential(nn.Conv2d(self.channels[2], self.channels[1], 1, 1, 0), nn.BatchNorm2d(self.channels[1]), nn.ReLU())
        self.conv_rgb3 = nn.Sequential(nn.Conv2d(self.channels[3], self.channels[1], 1, 1, 0), nn.BatchNorm2d(self.channels[1]), nn.ReLU())

        self.DWT = DWT()
        self.after_dwt0 = nn.BatchNorm2d(self.channels[1])
        self.relu = nn.LeakyReLU(0.2)
        self.conv2_3 = self._make_layer(BasicBlockL, self.channels[1], self.channels[1])
        self.conv2_4 = self._make_layer(BasicBlockH, self.channels[1]*3, self.channels[1]*3)

        self.up_1 = nn.Sequential(UpSampler(scale=2, n_feats=self.channels[1] * 3))
        self.up_2 = nn.Sequential(UpSampler(scale=4, n_feats=self.channels[1] * 3))
        self.up_3 = nn.Sequential(UpSampler(scale=2, n_feats=self.channels[1]))
        self.up_4 = nn.Sequential(UpSampler(scale=4, n_feats=self.channels[1]))

        self.fusion_h2 = nn.Conv2d(in_channels=3 * self.channels[1] * 3, out_channels=self.channels[1] * 3, kernel_size=1, stride=1, padding=0)
        self.fusion_h3 = nn.Conv2d(in_channels=self.channels[1] * 3, out_channels=self.channels[1], kernel_size=1,
                                   stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=self.channels[1] * 3, out_channels=self.channels[1], kernel_size=1, stride=1, padding=0)

        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32 + 1),
            nn.Sigmoid(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels[1], self.channels[1], 3, 1, 1),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(),
            nn.Conv2d(self.channels[1], 1, 1, 1, 0)
        )

        self.fuse = FAI(self.channels[1], num_heads=8, level=1)

        self.A2SP5 = CIU(self.channels[1], self.channels[1])
        self.A2SP4 = CIU(self.channels[1], self.channels[1])
        self.A2SP3 = CIU(self.channels[1], self.channels[1])
        self.A2SP2 = CIU(self.channels[1], self.channels[1])

        # Deep supervision heads
        if self.training:
            self.Sal_Head_2 = self._make_multi_head(self.channels[1])
            self.Sal_Head_3 = self._make_multi_head(self.channels[1])
            self.Sal_Head_4 = self._make_multi_head(self.channels[1])
            self.Sal_Head_5 = self._make_multi_head(self.channels[1])

        self.Sal_Head_Multi = self._make_multi_head(self.channels[1])

        # ============================================================
        # multi-scale CPC module
        # ============================================================
        CPCClass = CategoryPrototypeContrastAdaptiveTemp if adaptive_temp else CategoryPrototypeContrast

        def _make_cpc():
            if adaptive_temp:
                return CPCClass(
                    in_channels=self.channels[1],
                    num_classes=num_classes,
                    proto_dim=64,
                    base_temperature=temperature,
                    enable_refinement=enable_refinement
                )
            else:
                return CPCClass(
                    in_channels=self.channels[1],
                    num_classes=num_classes,
                    proto_dim=64,
                    temperature=temperature,
                    enable_refinement=enable_refinement
                )

        if shared_prototypes:
            shared_cpc = _make_cpc()
            self.cpc_f1 = shared_cpc
            self.cpc_f2 = shared_cpc
            self.cpc_f3 = shared_cpc
            self.cpc_f4 = shared_cpc
        else:
            self.cpc_f1 = _make_cpc()
            self.cpc_f2 = _make_cpc()
            self.cpc_f3 = _make_cpc()
            self.cpc_f4 = _make_cpc()

    def _make_layer(self, block, inplanes, out_planes):
        layers = []
        layers.append(block(inplanes, out_planes))
        return nn.Sequential(*layers)

    def _make_multi_head(self, channel):
        return nn.Sequential(
            nn.Conv2d(channel, channel // 2, 3, 1, 1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(),
            nn.Conv2d(channel // 2, self.num_classes, 1, 1, 0)
        )

    def forward(self, RGB, gt_labels=None):
        image_size = RGB.size()[2:]
        baseline_net = self.rgb_swin(RGB)

        # Encoder
        Fr1 = baseline_net[0]
        Fr2 = baseline_net[1]
        Fr3 = baseline_net[2]
        Fr4 = baseline_net[3]

        Fr1 = self.conv_rgb0(Fr1)
        Fr2 = self.conv_rgb1(Fr2)
        Fr3 = self.conv_rgb2(Fr3)
        Fr4 = self.conv_rgb3(Fr4)

        # DWT
        LL_2, LH_2, HL_2, HH_2 = self.DWT(Fr2)
        LL_2 = self.after_dwt0(LL_2)
        LH_2 = self.after_dwt0(LH_2)
        HL_2 = self.after_dwt0(HL_2)
        HH_2 = self.after_dwt0(HH_2)
        H_2 = torch.cat((LH_2, HL_2, HH_2), 1)
        L_2 = self.conv2_3(LL_2)
        H_2 = self.conv2_4(H_2)

        LL_3, LH_3, HL_3, HH_3 = self.DWT(Fr3)
        LL_3 = self.after_dwt0(LL_3)
        LH_3 = self.after_dwt0(LH_3)
        HL_3 = self.after_dwt0(HL_3)
        HH_3 = self.after_dwt0(HH_3)
        H_3 = torch.cat((LH_3, HL_3, HH_3), 1)
        L_3 = self.conv2_3(LL_3)
        H_3 = self.conv2_4(H_3)

        LL_4, LH_4, HL_4, HH_4 = self.DWT(Fr4)
        LL_4 = self.after_dwt0(LL_4)
        LH_4 = self.after_dwt0(LH_4)
        HL_4 = self.after_dwt0(HL_4)
        HH_4 = self.after_dwt0(HH_4)
        H_4 = torch.cat((LH_4, HL_4, HH_4), 1)
        L_4 = self.conv2_3(LL_4)
        H_4 = self.conv2_4(H_4)

        H_3 = self.up_1(H_3)
        H_4 = self.up_2(H_4)
        high2 = self.fusion_h2(torch.cat([H_3, H_4, H_2], dim=1))
        high = self.conv2(high2)

        L_3 = self.up_3(L_3)
        L_4 = self.up_4(L_4)
        low = self.fusion_h3(torch.cat([L_3, L_4, L_2], dim=1))

        F5 = self.fuse(high, low)

        # BA module
        bz = RGB.shape[0]
        rgb_gap = self.gap1(Fr1)
        rgb_gap = rgb_gap.view(bz, -1)
        feat = self.fc(rgb_gap)
        gate = feat[:, -1].view(bz, 1, 1, 1)
        edge = gate * high

        # decoding
        F4 = self.A2SP5(Fr4, F5, edge)
        F3 = self.A2SP4(Fr3 + F.interpolate(F4, Fr3.shape[2:], mode='bilinear', align_corners=False), F4, edge)
        F2 = self.A2SP3(Fr2 + F.interpolate(F4, Fr2.shape[2:], mode='bilinear', align_corners=False) +
                            F.interpolate(F3, Fr2.shape[2:], mode='bilinear', align_corners=False), F3, edge)
        F1 = self.A2SP2(Fr1 + F.interpolate(F4, Fr1.shape[2:], mode='bilinear', align_corners=False) +
                            F.interpolate(F3, Fr1.shape[2:], mode='bilinear', align_corners=False) +
                            F.interpolate(F2, Fr1.shape[2:], mode='bilinear', align_corners=False), F2, edge)

        # ============================================================
        # multi-scale CPC refinement
        # ============================================================
        cpc_losses = []

        F4, cpc_loss_4 = self.cpc_f4(F4, gt_labels)
        if cpc_loss_4 is not None:
            cpc_losses.append(cpc_loss_4)

        F3, cpc_loss_3 = self.cpc_f3(F3, gt_labels)
        if cpc_loss_3 is not None:
            cpc_losses.append(cpc_loss_3)

        F2, cpc_loss_2 = self.cpc_f2(F2, gt_labels)
        if cpc_loss_2 is not None:
            cpc_losses.append(cpc_loss_2)

        F1, cpc_loss_1 = self.cpc_f1(F1, gt_labels)
        if cpc_loss_1 is not None:
            cpc_losses.append(cpc_loss_1)

        # aggregate the multi-scale CPC losses
        total_cpc_loss = sum(cpc_losses) / max(len(cpc_losses), 1) if cpc_losses else None

        if self.training:
            F5_out = F.interpolate(self.Sal_Head_5(F5), image_size, mode='bilinear', align_corners=False)
            F4_out = F.interpolate(self.Sal_Head_4(F4), image_size, mode='bilinear', align_corners=False)
            F3_out = F.interpolate(self.Sal_Head_3(F3), image_size, mode='bilinear', align_corners=False)
            F2_out = F.interpolate(self.Sal_Head_2(F2), image_size, mode='bilinear', align_corners=False)
            F1_out = F.interpolate(self.Sal_Head_Multi(F1), image_size, mode='bilinear', align_corners=False)
            edge_out = F.interpolate(self.conv(edge), image_size, mode='bilinear', align_corners=False)

            return F1_out, F2_out, F3_out, F4_out, F5_out, edge_out, total_cpc_loss
        else:
            F_out = F.interpolate(self.Sal_Head_Multi(F1), image_size, mode='bilinear', align_corners=False)
            return F_out

    def load_from_single_class_weights(self, weight_path):
        print('Loading from single-class weights:', weight_path)
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)

        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        model_dict = self.state_dict()
        new_state_dict = {}
        loaded_count = 0
        skipped_count = 0
        for k, v in state_dict.items():
            if 'Sal_Head' in k or 'cpc' in k:
                skipped_count += 1
                continue
            if k in model_dict and model_dict[k].shape == v.shape:
                new_state_dict[k] = v
                loaded_count += 1
            else:
                skipped_count += 1

        model_dict.update(new_state_dict)
        self.load_state_dict(model_dict)
        print(f'Loaded {loaded_count} params, skipped {skipped_count}')
