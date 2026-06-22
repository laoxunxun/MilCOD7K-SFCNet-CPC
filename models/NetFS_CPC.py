"""
A Frequency-Sensitive CPC version of the SFCNet multi-class segmentation network

Key differences from NetMultiClass:
- NetMultiClass: CPC only looks at the mixed decoder-end feature F1
- NetFS_CPC:     FS-CPC uses both the HF/LF features after DWT decomposition and the decoder features

Architecture data flow:
  Input → SMT-Tiny → Channel Align → DWT → BasicBlockL/H
  -> multi-scale fusion -> high (HF fused), low (LF fused)
  → FAI(high, low) → EA → CIU×4 → F1
  -> FS-CPC(high, low, F1) -> refined_F1 -> multi-class segmentation head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import (
    Conv, UpSampler, FAI, DWT, BasicBlockL, BasicBlockH,
    CIU, FrequencySensitiveCPC
)
from models.smt import smt_t


class NetFS_CPC(nn.Module):
    """
    Frequency-Sensitive CPC multi-class segmentation network

    FS-CPC takes three inputs:
    1. high - DWT HF fused feature (texture/edge info)
    2. low - DWT LF fused feature (structure/context info)
    3. F1 - the decoder's final feature

    Build class prototypes in the HF and LF spaces separately and fuse them via an adaptive gate.
    """

    def __init__(self, num_classes=5, temperature=0.07, enable_refinement=True,
                 proto_dim=64, hf_temperature=None, lf_temperature=None):
        super(NetFS_CPC, self).__init__()

        self.num_classes = num_classes
        self.temperature = temperature
        self.enable_refinement = enable_refinement
        channels = [64, 128, 256, 512]
        self.channels = channels

        # ==================== Encoder (identical to the original SFCNet) ====================
        self.rgb_swin = smt_t()

        # channel alignment
        self.conv_rgb0 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 1, 1, 0),
            nn.BatchNorm2d(channels[1]), nn.ReLU()
        )
        self.conv_rgb1 = nn.Sequential(
            nn.Conv2d(channels[1], channels[1], 1, 1, 0),
            nn.BatchNorm2d(channels[1]), nn.ReLU()
        )
        self.conv_rgb2 = nn.Sequential(
            nn.Conv2d(channels[2], channels[1], 1, 1, 0),
            nn.BatchNorm2d(channels[1]), nn.ReLU()
        )
        self.conv_rgb3 = nn.Sequential(
            nn.Conv2d(channels[3], channels[1], 1, 1, 0),
            nn.BatchNorm2d(channels[1]), nn.ReLU()
        )

        # DWT + frequency branch processing
        self.DWT = DWT()
        self.after_dwt0 = nn.BatchNorm2d(channels[1])
        self.relu = nn.LeakyReLU(0.2)
        self.conv2_3 = self._make_layer(BasicBlockL, channels[1], channels[1])
        self.conv2_4 = self._make_layer(BasicBlockH, channels[1] * 3, channels[1] * 3)

        # upsampling
        self.up_1 = nn.Sequential(UpSampler(scale=2, n_feats=channels[1] * 3))
        self.up_2 = nn.Sequential(UpSampler(scale=4, n_feats=channels[1] * 3))
        self.up_3 = nn.Sequential(UpSampler(scale=2, n_feats=channels[1]))
        self.up_4 = nn.Sequential(UpSampler(scale=4, n_feats=channels[1]))

        # frequency fusion
        self.fusion_h2 = nn.Conv2d(
            in_channels=3 * channels[1] * 3, out_channels=channels[1] * 3,
            kernel_size=1, stride=1, padding=0
        )
        self.fusion_h3 = nn.Conv2d(
            in_channels=channels[1] * 3, out_channels=channels[1],
            kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=channels[1] * 3, out_channels=channels[1],
            kernel_size=1, stride=1, padding=0
        )

        # ==================== FAI + BA + Decoder (same as the original SFCNet) ====================
        # BA edge awareness
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32 + 1),
            nn.Sigmoid(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels[1], channels[1], 3, 1, 1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], 1, 1, 1, 0)
        )

        # FAI frequency-adaptive fusion
        self.fuse = FAI(channels[1], num_heads=8, level=1)

        # CIU decoder
        self.A2SP5 = CIU(channels[1], channels[1])
        self.A2SP4 = CIU(channels[1], channels[1])
        self.A2SP3 = CIU(channels[1], channels[1])
        self.A2SP2 = CIU(channels[1], channels[1])

        # ==================== [newly added] FS-CPC module ====================
        self.fscpc = FrequencySensitiveCPC(
            in_channels=channels[1],
            num_classes=num_classes,
            proto_dim=proto_dim,
            temperature=self.temperature,
            enable_refinement=self.enable_refinement,
            hf_temperature=hf_temperature,
            lf_temperature=lf_temperature
        )

        # ==================== segmentation head ====================
        if self.training:
            self.Sal_Head_2 = self._make_multi_head(channels[1])
            self.Sal_Head_3 = self._make_multi_head(channels[1])
            self.Sal_Head_4 = self._make_multi_head(channels[1])
            self.Sal_Head_5 = self._make_multi_head(channels[1])

        self.Sal_Head_Multi = self._make_multi_head(channels[1])

    def _make_layer(self, block, inplanes, out_planes):
        layers = []
        layers.append(block(inplanes, out_planes))
        return nn.Sequential(*layers)

    def _make_multi_head(self, channel):
        """Create the multi-class segmentation head"""
        return nn.Sequential(
            nn.Conv2d(channel, channel // 2, 3, 1, 1),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(),
            nn.Conv2d(channel // 2, self.num_classes, 1, 1, 0)
        )

    def forward(self, RGB, gt_labels=None):
        image_size = RGB.size()[2:]
        baseline_net = self.rgb_swin(RGB)

        # ==================== Encoder ====================
        Fr1 = baseline_net[0]
        Fr2 = baseline_net[1]
        Fr3 = baseline_net[2]
        Fr4 = baseline_net[3]

        # channel alignment
        Fr1 = self.conv_rgb0(Fr1)
        Fr2 = self.conv_rgb1(Fr2)
        Fr3 = self.conv_rgb2(Fr3)
        Fr4 = self.conv_rgb3(Fr4)

        # ==================== DWT + frequency branch processing ====================
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

        # multi-scale frequency fusion
        H_3 = self.up_1(H_3)
        H_4 = self.up_2(H_4)
        high2 = self.fusion_h2(torch.cat([H_3, H_4, H_2], dim=1))
        high = self.conv2(high2)  # <- HF fused feature (B, 128, H/8, W/8)

        L_3 = self.up_3(L_3)
        L_4 = self.up_4(L_4)
        low = self.fusion_h3(torch.cat([L_3, L_4, L_2], dim=1))  # <- LF fused feature (B, 128, H/8, W/8)

        # ==================== FAI fusion ====================
        F5 = self.fuse(high, low)

        # ==================== BA edge awareness ====================
        bz = RGB.shape[0]
        rgb_gap = self.gap1(Fr1)
        rgb_gap = rgb_gap.view(bz, -1)
        feat = self.fc(rgb_gap)
        gate = feat[:, -1].view(bz, 1, 1, 1)
        edge = gate * high

        # ==================== Decoder (CIU×4) ====================
        F4 = self.A2SP5(Fr4, F5, edge)
        F3 = self.A2SP4(
            Fr3 + F.interpolate(F4, Fr3.shape[2:], mode='bilinear', align_corners=False),
            F4, edge
        )
        F2 = self.A2SP3(
            Fr2 + F.interpolate(F4, Fr2.shape[2:], mode='bilinear', align_corners=False)
                 + F.interpolate(F3, Fr2.shape[2:], mode='bilinear', align_corners=False),
            F3, edge
        )
        F1 = self.A2SP2(
            Fr1 + F.interpolate(F4, Fr1.shape[2:], mode='bilinear', align_corners=False)
                 + F.interpolate(F3, Fr1.shape[2:], mode='bilinear', align_corners=False)
                 + F.interpolate(F2, Fr1.shape[2:], mode='bilinear', align_corners=False),
            F2, edge
        )

        # ==================== [key change] FS-CPC frequency-aware prototype contrast ====================
        # pass in the three features: high, low, F1
        cpc_losses = None
        F1, cpc_losses = self.fscpc(high, low, F1, gt_labels)

        # ==================== output ====================
        if self.training:
            F5_out = F.interpolate(self.Sal_Head_5(F5), image_size, mode='bilinear', align_corners=False)
            F4_out = F.interpolate(self.Sal_Head_4(F4), image_size, mode='bilinear', align_corners=False)
            F3_out = F.interpolate(self.Sal_Head_3(F3), image_size, mode='bilinear', align_corners=False)
            F2_out = F.interpolate(self.Sal_Head_2(F2), image_size, mode='bilinear', align_corners=False)
            F1_out = F.interpolate(self.Sal_Head_Multi(F1), image_size, mode='bilinear', align_corners=False)
            edge_out = F.interpolate(self.conv(edge), image_size, mode='bilinear', align_corners=False)

            return F1_out, F2_out, F3_out, F4_out, F5_out, edge_out, cpc_losses
        else:
            F_out = F.interpolate(self.Sal_Head_Multi(F1), image_size, mode='bilinear', align_corners=False)
            return F_out

    def load_pre(self, pre_model):
        """Load pretrained backbone weights"""
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"NetFS_CPC loading pre_model: {pre_model}")

    def load_from_single_class_weights(self, weight_path):
        """
        Load from single-class SFCNet weights, reusing the backbone and encoder
        The FS-CPC module and the segmentation head are randomly initialized
        """
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
            # skip the segmentation head
            if 'Sal_Head' in k:
                skipped_count += 1
                continue
            # skip the CPC/FS-CPC modules (not present in the original model)
            if 'cpc' in k or 'fscpc' in k:
                skipped_count += 1
                continue
            if k in model_dict and model_dict[k].shape == v.shape:
                new_state_dict[k] = v
                loaded_count += 1
            else:
                skipped_count += 1

        model_dict.update(new_state_dict)
        self.load_state_dict(model_dict)

        print(f'Weights loaded! Loaded {loaded_count} params, skipped {skipped_count} params.')
        print('FS-CPC module and all segmentation heads randomly initialized.')

    def load_from_multiclass_weights(self, weight_path):
        """
        Load from NetMultiClass weights (reuse backbone + encoder + segmentation head)
        Only the FS-CPC module is randomly initialized
        """
        print('Loading from multi-class weights:', weight_path)
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
            # skip the CPC module (replaced by FS-CPC)
            if 'cpc' in k and 'fscpc' not in k:
                skipped_count += 1
                continue
            # skip the FS-CPC module (new structure, does not match)
            if 'fscpc' in k:
                skipped_count += 1
                continue
            if k in model_dict and model_dict[k].shape == v.shape:
                new_state_dict[k] = v
                loaded_count += 1
            else:
                skipped_count += 1

        model_dict.update(new_state_dict)
        self.load_state_dict(model_dict)

        print(f'Weights loaded! Loaded {loaded_count} params, skipped {skipped_count} params.')
        print('FS-CPC module randomly initialized.')
