"""
SINet-V2 multi-class segmentation adaptation

Core idea:
    All intermediate features of the original SINet-V2 collapse to 1 channel and cannot distinguish semantic classes.
    This scheme obtains the 32-channel multi-scale features from the RFB module via a forward hook,
    Then build an FPN-style multi-class segmentation decoder on top of it.

    Keep the original NCD+GRA path as auxiliary binary supervision to help the backbone learn
    Better camouflaged-target features.

Architecture sketch:

    Res2Net backbone
        ↓
    RFB modules → x2_rfb(32ch, 44×44)  ──→ lateral2 ──┐
                 -> x3_rfb(32ch, 22x22)  --> lateral3 -->| FPN fusion
                 → x4_rfb(32ch, 11×11)  ──→ lateral4 ──┘
        ↓                                              ↓
    NCD + GRA (original path)                    seg_head -> 5-channel output
        ↓                                     ↓
    4 single-channel auxiliary outputs (for training)        main output (for inference)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SINetV2MultiClass(nn.Module):
    """
    The adapted SINet-V2 multi-class segmentation model.

    Capture the backbone's RFB features via a hook and build an FPN multi-class decoder.
    The 1-channel output of the original NCD/GRA path serves as an auxiliary loss during training.

    Args:
        base_model: the original SINet-V2 Network instance
        num_classes: number of segmentation classes (default 5)
        channel: number of RFB feature channels (default 32)
    """

    def __init__(self, base_model, num_classes=5, channel=32):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.channel = channel

        # ============================================================
        # Hook mechanism: capture the 32-channel output of the RFB module
        # ============================================================
        self._rfb_features = {}
        for name, module in base_model.named_modules():
            if name == 'rfb2_1':
                module.register_forward_hook(self._make_hook('rfb2'))
            elif name == 'rfb3_1':
                module.register_forward_hook(self._make_hook('rfb3'))
            elif name == 'rfb4_1':
                module.register_forward_hook(self._make_hook('rfb4'))

        # ============================================================
        # FPN multi-class decoder
        # ============================================================
        # lateral connections: unify the channel count
        self.lateral4 = nn.Conv2d(channel, channel, 1)  # 11×11
        self.lateral3 = nn.Conv2d(channel, channel, 1)  # 22×22
        self.lateral2 = nn.Conv2d(channel, channel, 1)  # 44×44

        # smoothing convolutions
        self.smooth3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.smooth2 = nn.Conv2d(channel, channel, 3, padding=1)

        # semantic enhancement module (makes the features more semantically discriminative)
        self.context = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        # final segmentation head
        self.seg_head = nn.Conv2d(channel, num_classes, 1)

        # initialize the newly added layers
        self._init_weights()

    def _make_hook(self, name):
        """Create a forward hook that saves the RFB features"""
        def hook(module, input, output):
            self._rfb_features[name] = output
        return hook

    def _init_weights(self):
        """Initialize the weights of the FPN decoder"""
        for m in [self.lateral4, self.lateral3, self.lateral2,
                  self.smooth3, self.smooth2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        nn.init.kaiming_normal_(self.seg_head.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.seg_head.bias)

        for m in self.context.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def fpn_decode(self, f2, f3, f4):
        """
        FPN top-down fusion.

        Args:
            f2: (B, C, 44, 44) - shallowest layer, highest spatial resolution
            f3: (B, C, 22, 22) - middle layer
            f4: (B, C, 11, 11) - deepest layer, richest semantics

        Returns:
            multi_class_logits: (B, num_classes, 44, 44)
        """
        # top-down fusion
        p4 = self.lateral4(f4)                          # (B, C, 11, 11)
        p3 = self.lateral3(f3) + F.interpolate(
            p4, size=f3.shape[2:], mode='bilinear', align_corners=False)  # (B, C, 22, 22)
        p3 = self.smooth3(p3)

        p2 = self.lateral2(f2) + F.interpolate(
            p3, size=f2.shape[2:], mode='bilinear', align_corners=False)  # (B, C, 44, 44)
        p2 = self.smooth2(p2)

        # semantic enhancement
        p2 = self.context(p2)

        # segmentation output
        logits = self.seg_head(p2)  # (B, num_classes, 44, 44)
        return logits

    def forward(self, x):
        """
        Forward pass.

        Returns:
            outputs: list of tensors
                - outputs[0]: main multi-class segmentation output (B, num_classes, H, W)
                - outputs[1:5]: the 4 auxiliary outputs of the original SINet-V2 (each B, 1, H, W)
        """
        # clear the hook cache
        self._rfb_features.clear()

        # run the original SINet-V2 (also triggering the hook to capture RFB features)
        binary_outputs = self.base_model(x)  # 4 single-channel outputs

        # get the RFB features
        f2 = self._rfb_features['rfb2']  # (B, 32, 44, 44)
        f3 = self._rfb_features['rfb3']  # (B, 32, 22, 22)
        f4 = self._rfb_features['rfb4']  # (B, 32, 11, 11)

        # FPN multi-class decoding
        mc_logits = self.fpn_decode(f2, f3, f4)  # (B, 5, 44, 44)

        # upsample to the input size
        mc_logits = F.interpolate(mc_logits, size=x.shape[2:],
                                  mode='bilinear', align_corners=False)

        # return: multi-class main output + the original binary auxiliary output
        return [mc_logits] + list(binary_outputs)
