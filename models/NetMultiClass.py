"""
Multi-class segmentation version of SFCNet
Support semantic segmentation of N classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import Conv, UpSampler, FAI, DWT, BasicBlockL, BasicBlockH, CIU, CategoryPrototypeContrast
from models.smt import smt_t


class NetMultiClass(nn.Module):
    """
    Multi-class version of SFCNet
    Output N channels; each channel is the segmentation probability of one class
    """
    def __init__(self, num_classes=4, temperature=0.07, enable_refinement=True):
        super(NetMultiClass, self).__init__()

        self.num_classes = num_classes
        self.temperature = temperature
        self.enable_refinement = enable_refinement
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
            nn.Conv2d(self.channels[1], 1, 1, 1, 0)  # edge detection is still single-channel
        )

        self.fuse = FAI(self.channels[1], num_heads=8, level=1)

        self.A2SP5 = CIU(self.channels[1], self.channels[1])
        self.A2SP4 = CIU(self.channels[1], self.channels[1])
        self.A2SP3 = CIU(self.channels[1], self.channels[1])
        self.A2SP2 = CIU(self.channels[1], self.channels[1])

        # during training, also output the stage2-5 results (multi-class)
        if self.training:
            self.Sal_Head_2 = self._make_multi_head(self.channels[1])
            self.Sal_Head_3 = self._make_multi_head(self.channels[1])
            self.Sal_Head_4 = self._make_multi_head(self.channels[1])
            self.Sal_Head_5 = self._make_multi_head(self.channels[1])

        # [key change] multi-class segmentation head
        # output num_classes channels; each channel is the logits of one class
        self.Sal_Head_Multi = self._make_multi_head(self.channels[1])

        # [new] CPC category prototype contrast module - enhances inter-class discrimination of decoder features
        self.cpc = CategoryPrototypeContrast(
            in_channels=self.channels[1],
            num_classes=num_classes,
            proto_dim=64,
            temperature=self.temperature,
            enable_refinement=self.enable_refinement
        )

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

        # Encoder - the backbone extracts features
        Fr1 = baseline_net[0]
        Fr2 = baseline_net[1]
        Fr3 = baseline_net[2]
        Fr4 = baseline_net[3]

        # channel alignment
        Fr1 = self.conv_rgb0(Fr1)
        Fr2 = self.conv_rgb1(Fr2)
        Fr3 = self.conv_rgb2(Fr3)
        Fr4 = self.conv_rgb3(Fr4)

        # DWT processing
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

        # [new] CPC category prototype contrast refinement
        cpc_loss = None
        F1, cpc_loss = self.cpc(F1, gt_labels)

        if self.training:
            # during training, output multi-class results of multiple stages + edge + cpc_loss
            F5_out = F.interpolate(self.Sal_Head_5(F5), image_size, mode='bilinear', align_corners=False)
            F4_out = F.interpolate(self.Sal_Head_4(F4), image_size, mode='bilinear', align_corners=False)
            F3_out = F.interpolate(self.Sal_Head_3(F3), image_size, mode='bilinear', align_corners=False)
            F2_out = F.interpolate(self.Sal_Head_2(F2), image_size, mode='bilinear', align_corners=False)
            F1_out = F.interpolate(self.Sal_Head_Multi(F1), image_size, mode='bilinear', align_corners=False)
            edge_out = F.interpolate(self.conv(edge), image_size, mode='bilinear', align_corners=False)

            return F1_out, F2_out, F3_out, F4_out, F5_out, edge_out, cpc_loss
        else:
            # at inference, output only the multi-class segmentation result
            F_out = F.interpolate(self.Sal_Head_Multi(F1), image_size, mode='bilinear', align_corners=False)
            return F_out

    def load_pre(self, pre_model):
        """Load pretrained weights"""
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"NetMultiClass loading pre_model: {pre_model}")

    def load_from_single_class_weights(self, weight_path):
        """
        Load from single-class SFCNet weights to initialize the multi-class model
        Copy the backbone and encoder, reinitialize all segmentation heads
        """
        print('Loading from single-class weights:', weight_path)
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)

        # get the state_dict
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # get the current model's state_dict
        model_dict = self.state_dict()

        # filter and match the weights
        new_state_dict = {}
        loaded_count = 0
        skipped_count = 0
        for k, v in state_dict.items():
            # skip all segmentation-head weights (single-class vs multi-class channel mismatch)
            if 'Sal_Head' in k:
                skipped_count += 1
                continue
            # skip the CPC module weights (not present in the original model)
            if 'cpc' in k:
                skipped_count += 1
                continue
            # copy all other weights (backbone, DWT, CIU, FAI, etc.)
            if k in model_dict and model_dict[k].shape == v.shape:
                new_state_dict[k] = v
                loaded_count += 1
            else:
                skipped_count += 1

        # load the matching weights
        model_dict.update(new_state_dict)
        self.load_state_dict(model_dict)

        print(f'Weights loaded! Loaded {loaded_count} params, skipped {skipped_count} params.')
        print('All segmentation heads randomly initialized (including intermediate deep-supervision heads).')
