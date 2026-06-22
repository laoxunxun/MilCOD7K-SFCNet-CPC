import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

    def initialize(self):
        weight_init(self)

def weight_init(module):
    for n, m in module.named_children():
      #  print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Sequential,nn.ModuleList,nn.ModuleDict)):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (LayerNorm,nn.ReLU,nn.ReLU,nn.AdaptiveAvgPool2d,nn.Softmax,nn.AvgPool2d)):
            pass
        else:
            m.initialize()

#----------------------------------------DTF--------------------------------------------------------
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh

class BasicBlockL(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1, norm_layer=None):
        super(BasicBlockL, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                  padding=1, groups=groups)
        self.conv1_1 = nn.Conv2d(inplanes, planes, kernel_size=1, dilation=3)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2)

        self.conv2 =  nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                  padding=1, groups=groups)
        self.conv2_1 = nn.Conv2d(inplanes, planes, kernel_size=1, dilation=3)
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out1 = self.conv1_1(out1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 += identity
        out2 = self.conv2(out1)
        out2 = self.conv2_1(out2)
        out2 = self.bn2(out2)
        out = self.relu(out2)

        return out

class  BasicBlockH(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1, norm_layer=None):
        super(BasicBlockH, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.conv1_1 = nn.Conv2d(inplanes, planes, kernel_size=(1,3), padding=(0, 1))
        self.conv1_2 = nn.Conv2d(inplanes, planes, kernel_size=(3,1), padding=(1, 0))
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.conv2_1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 3), padding=(0, 1))
        self.conv2_2 = nn.Conv2d(inplanes, planes, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = norm_layer(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out1 = self.conv1(x)
        out1 = self.conv1_1(out1)
        out1 = self.conv1_2(out1)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out = identity + out1

        out = self.conv2(out)
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class UpSampler(nn.Sequential):     # upsampling
    def __init__(self, scale, n_feats):

        m = []
        if scale == 8:
            kernel_size = 3
        elif scale == 16:
            kernel_size = 5
        else:
            kernel_size = 1

        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(in_channels=n_feats, out_channels=4 * n_feats, kernel_size=kernel_size, stride=1,
                                   padding=kernel_size // 2))
                m.append(nn.PixelShuffle(upscale_factor=2))
                m.append(nn.PReLU())
        super(UpSampler, self).__init__(*m)

class ConvBNReLu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, padding=1, dilation=1):
        super(ConvBNReLu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=1,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = LayerNorm(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def initialize(self):
        weight_init(self)

def window_partition(x, window_size):
    # input B C H W
    x = x.permute(0, 2, 3, 1)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows  # B_ H_ W_ C

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x.permute(0, 3, 1, 2)

class MLP(nn.Module):
    def __init__(self, inchannel, outchannel, bias=False):
        super(MLP, self).__init__()
        self.conv1 = nn.Linear(inchannel, outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.ln = nn.LayerNorm(outchannel)

    def forward(self, x):
        return self.relu(self.ln(self.conv1(x)) + x)

    def initialize(self):
        weight_init(self)

class FAI(nn.Module):  # x hf  y  lf
    def __init__(self, dim, num_heads=8, level=8, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.level = level
        self.mul = nn.Sequential(ConvBNReLu(dim, dim), ConvBNReLu(dim, dim, kernel_size=1, padding=0))
        self.add = nn.Sequential(ConvBNReLu(dim, dim), ConvBNReLu(dim, dim, kernel_size=1, padding=0))

        self.conv_x = nn.Sequential(ConvBNReLu(dim, dim), ConvBNReLu(dim, dim, kernel_size=1, padding=0))

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)

        self.lnx = nn.LayerNorm(dim)
        self.lny = nn.LayerNorm(dim)
        self.ln = nn.LayerNorm(dim)

        self.shortcut = nn.Linear(dim, dim)

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            LayerNorm(dim)
        )
        self.mlp = MLP(dim, dim)

    def forward(self, x, y):
        origin_size = x.shape[2]
        ws = origin_size // self.level // 4
        x = self.conv_x(x)

        x = window_partition(x, ws)
        y = window_partition(y, ws)

        x = x.view(x.shape[0], -1, x.shape[3])
        sc1 = x
        x = self.lnx(x)
        y = y.view(y.shape[0], -1, y.shape[3])
        y = self.lny(y)
        B, N, C = x.shape
        y_kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q = x_q[0]
        y_k, y_v = y_kv[0], y_kv[1]
        attn = (x_q @ y_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ y_v).transpose(1, 2).reshape(B, N, C)
        x = self.act(x + sc1)
        x = self.act(x + self.mlp(x))
        x = x.view(-1, ws, ws, C)
        x = window_reverse(x, ws, origin_size, origin_size)
        x = self.act(self.conv2(x) + x)
        return x

    def initialize(self):
        weight_init(self)
# ----------------------------------------Decoder----------------------------------------
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MFE(nn.Module):
    def __init__(self, inplanes, planes):
        super(MFE, self).__init__()
        self.conv_rgb0 = nn.Sequential(nn.Conv2d(inplanes, planes, 1, 1, 0),
                                       nn.BatchNorm2d(planes), nn.ReLU())
        self.conv_1 = nn.Sequential(nn.Conv2d(planes, planes, 3, dilation=1, padding=1),
                                       nn.BatchNorm2d(planes), nn.ReLU())
        self.conv_2 = nn.Sequential(nn.Conv2d(planes, planes, 3, dilation=3, padding=3),
                                       nn.BatchNorm2d(planes), nn.ReLU())
        self.conv_3 = nn.Sequential(nn.Conv2d(planes, planes, 3, dilation=5, padding=5),
                                       nn.BatchNorm2d(planes), nn.ReLU())
        self.conv_4 = nn.Sequential(nn.Conv2d(planes, planes, 3, dilation=7, padding=7),
                                       nn.BatchNorm2d(planes), nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=1), nn.PReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * planes, planes, kernel_size=1), nn.BatchNorm2d(planes), nn.PReLU()
        )
        self.up = nn.Sequential(*UpSampler(scale=4, n_feats=inplanes))
    def forward(self, f):
        f_0 = self.conv_rgb0(f)
        f_1 = self.conv_1(f_0)
        f_2 = self.conv_2(f_0)
        f_3 = self.conv_3(f_0)
        f_4 = self.conv_4(f_0)
        f_6 = self.fuse(torch.cat((f_1, f_2, f_3, f_4), 1))
        return f_6

class Attention(nn.Module):
    def __init__(self, dim=64,out=64,  num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, out, kernel_size=1, bias=bias)

    def forward(self, x, y):    # (fre, spa)
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)     
        k = torch.nn.functional.normalize(k, dim=-1)    

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out) + x    
        return out

class CIU(nn.Module):

    def __init__(self, inplanes, planes):
        super(CIU, self).__init__()
        self.GAP_Conv = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, inplanes, 1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )

        self.conv_5 = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1)
        self.att1 = Attention( dim=inplanes, out=inplanes)
        self.att = MFE(inplanes, planes)
        self.ra_conv1 = BasicConv2d(planes + planes, planes, kernel_size=3, padding=1)

    def forward(self, x, F5, edge):

        out = self.att(x)
        if F5.size()[2:] != x.size()[2:]:
            F5 = F.interpolate(F5, size=x.size()[2:], mode='nearest')
            edge = F.interpolate(edge, size=x.size()[2:], mode='nearest')

        v_attention = self.GAP_Conv(x)
        out_1 = self.att1(F5, x)
        y = out_1 * out * v_attention
        y = self.conv(y)
        y = self.ra_conv1(torch.cat((y, edge), dim=1))
        return y

#------------------------------------------------Decoder--------------------------------------------------------------
class Conv(nn.Module):
    def __init__(self, channel):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, 1, 1, 1, 0)
        )

    def forward(self, x):
        y = self.conv(x)
        return y


#------------------------------------------------CPC Module--------------------------------------------------------------
class CategoryPrototypeContrast(nn.Module):
    """
    Category Prototype Contrast module (Category Prototype Contrast, CPC)
    For the end of the multi-class segmentation decoder; enhances inter-class discrimination via learnable class prototypes.

    Core idea:
    1. project the decoder features into a low-dimensional prototype space
    2. compute the similarity of each pixel to the K class prototypes
    3. generate refined features via similarity-weighted prototypes, added residually to the original features
    4. additionally provide an InfoNCE contrastive loss during training
    """

    def __init__(self, in_channels=128, num_classes=5, proto_dim=64, temperature=0.07, enable_refinement=True):
        super(CategoryPrototypeContrast, self).__init__()
        self.num_classes = num_classes
        self.proto_dim = proto_dim
        self.temperature = temperature
        self.enable_refinement = enable_refinement

        # learnable class prototype vectors (K, D)
        # an anchor for each class in the embedding space
        self.prototypes = nn.Parameter(torch.randn(num_classes, proto_dim))
        nn.init.xavier_uniform_(self.prototypes.unsqueeze(0))

        # Step 1: feature projection C -> D
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proto_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proto_dim),
            nn.ReLU(inplace=True)
        )

        # Step 4: residual reconstruction D -> C
        self.rebuild = nn.Sequential(
            nn.Conv2d(proto_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x, gt_labels=None):
        """
        Args:
            x: decoder feature (B, C, H, W)
            gt_labels: ground-truth labels (B, H_gt, W_gt) int, used only during training (auto-downsampled to feature size)
        Returns:
            refined: refined feature (B, C, H, W)
            contrastive_loss: contrastive loss (has a value during training, None at inference)
        """
        B, C, H, W = x.shape

        # downsample gt_labels to the feature-map size
        if gt_labels is not None:
            gt_H, gt_W = gt_labels.shape[1], gt_labels.shape[2]
            if gt_H != H or gt_W != W:
                gt_labels = F.interpolate(
                    gt_labels.unsqueeze(1).float(), size=(H, W),
                    mode='nearest'
                ).squeeze(1).long()

        # Step 1: project into the prototype space
        q = self.proj(x)  # (B, D, H, W)

        # Step 2: compute the pixel-prototype cosine similarity
        # q: (B, D, H*W), prototypes: (K, D)
        q_flat = q.flatten(2)  # (B, D, N) where N=H*W
        q_norm = F.normalize(q_flat, dim=1)  # L2 normalize along channel
        p_norm = F.normalize(self.prototypes, dim=-1)  # (K, D)

        # (B, D, N) x (D, K) -> (B, N, K)
        # p_norm: (K, D) -> transpose -> (D, K) for matmul
        sim = torch.bmm(q_norm.transpose(1, 2), p_norm.t().unsqueeze(0).expand(B, -1, -1))
        sim = sim.permute(0, 2, 1).view(B, self.num_classes, H, W)  # (B, K, H, W)

        # Step 3: class-affinity-weighted refinement
        attn = F.softmax(sim / self.temperature, dim=1)  # (B, K, H, W)

        # each pixel = an affinity-weighted combination of the class prototypes
        # attn: (B, K, N), prototypes: (K, D) -> refined: (B, D, N)
        attn_flat = attn.flatten(2)  # (B, K, N)
        proto_feat = torch.bmm(p_norm.unsqueeze(0).expand(B, -1, -1).transpose(1, 2), attn_flat)
        # p_norm expanded: (B, K, D).T -> (B, D, K) x (B, K, N) -> (B, D, N)
        proto_feat = proto_feat.view(B, self.proto_dim, H, W)

        # Step 4: residual reconstruction (can be disabled with enable_refinement=False for ablation)
        if self.enable_refinement:
            refined = x + self.rebuild(proto_feat)
        else:
            refined = x

        # contrastive loss (computed only during training; reuses the already-computed sim to avoid duplicate bmm)
        contrastive_loss = None
        if self.training and gt_labels is not None:
            # sim: (B, K, H, W) -> (B, N, K) -> (B*N, K)
            sim_scaled = sim.permute(0, 2, 3, 1).reshape(B * H * W, self.num_classes) / self.temperature
            labels = gt_labels.flatten()  # (B*N,)
            contrastive_loss = F.cross_entropy(sim_scaled, labels)

        return refined, contrastive_loss


class CategoryPrototypeContrastAdaptiveTemp(nn.Module):
    """
    Adaptive-temperature CPC: each class learns an independent temperature parameter
    Hard classes (e.g. Tank) can use a sharper distribution to improve separation
    """

    def __init__(self, in_channels=128, num_classes=5, proto_dim=64, base_temperature=0.07, enable_refinement=True):
        super(CategoryPrototypeContrastAdaptiveTemp, self).__init__()
        self.num_classes = num_classes
        self.proto_dim = proto_dim
        self.base_temperature = base_temperature
        self.enable_refinement = enable_refinement

        # learnable class prototypes
        self.prototypes = nn.Parameter(torch.randn(num_classes, proto_dim))
        nn.init.xavier_uniform_(self.prototypes.unsqueeze(0))

        # adaptive temperature: one log_temperature per class, kept positive via exp
        self.log_temperatures = nn.Parameter(torch.zeros(num_classes))

        # projection & reconstruction
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proto_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proto_dim),
            nn.ReLU(inplace=True)
        )
        self.rebuild = nn.Sequential(
            nn.Conv2d(proto_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def get_temperatures(self):
        """Get the adaptive temperature (kept positive and not exceeding base_temperature)"""
        return self.base_temperature * torch.sigmoid(self.log_temperatures)

    def forward(self, x, gt_labels=None):
        B, C, H, W = x.shape

        if gt_labels is not None:
            gt_H, gt_W = gt_labels.shape[1], gt_labels.shape[2]
            if gt_H != H or gt_W != W:
                gt_labels = F.interpolate(
                    gt_labels.unsqueeze(1).float(), size=(H, W),
                    mode='nearest'
                ).squeeze(1).long()

        q = self.proj(x)
        q_flat = q.flatten(2)
        q_norm = F.normalize(q_flat, dim=1)
        p_norm = F.normalize(self.prototypes, dim=-1)

        sim = torch.bmm(q_norm.transpose(1, 2), p_norm.t().unsqueeze(0).expand(B, -1, -1))
        sim = sim.permute(0, 2, 1).view(B, self.num_classes, H, W)

        # use the adaptive temperature
        temps = self.get_temperatures()  # (K,)
        # sim / temps: need broadcasting (B, K, H, W) / (1, K, 1, 1)
        attn = F.softmax(sim / temps.view(1, -1, 1, 1), dim=1)

        attn_flat = attn.flatten(2)
        proto_feat = torch.bmm(p_norm.unsqueeze(0).expand(B, -1, -1).transpose(1, 2), attn_flat)
        proto_feat = proto_feat.view(B, self.proto_dim, H, W)

        if self.enable_refinement:
            refined = x + self.rebuild(proto_feat)
        else:
            refined = x

        contrastive_loss = None
        if self.training and gt_labels is not None:
            # the contrastive loss also uses the adaptive temperature
            sim_scaled = sim.permute(0, 2, 3, 1).reshape(B * H * W, self.num_classes)
            # scale each pixel by the temperature of its GT class
            labels = gt_labels.flatten()
            pixel_temps = temps[labels]  # (B*N,)
            sim_scaled = sim_scaled / pixel_temps.unsqueeze(1)  # scale
            contrastive_loss = F.cross_entropy(sim_scaled, labels)

        return refined, contrastive_loss


class FrequencySensitiveCPC(nn.Module):
    """
    Frequency-sensitive prototype contrastive learning module (Frequency-Sensitive Category Prototype Contrast, FS-CPC)

    Core idea:
    Build class prototypes in the HF and LF feature spaces separately, captured via cross-frequency contrastive learning
    Differentiated separability of different camouflaged classes in the frequency domain.

    Design motivation:
    - tanks and vehicles share camouflage paint -> high similarity in the HF prototype space (indistinguishable)
    - but tanks have a gun barrel and vehicles do not -> low similarity in the LF prototype space (distinguishable)
    - soldiers blend with the background in LF structure -> but distinguishable via HF texture breaks
    -> different classes have differentiated discriminative information across frequencies

    Differences from standard CPC:
    - standard CPC builds a single prototype in the mixed feature space (cannot distinguish frequency sources)
    - FS-CPC models the HF/LF prototype spaces separately and fuses them via an adaptive gate
    """

    def __init__(self, in_channels=128, num_classes=5, proto_dim=64,
                 temperature=0.07, enable_refinement=True,
                 hf_temperature=None, lf_temperature=None,
                 alignment_weight=0.1, gate_diversity_weight=0.05):
        super(FrequencySensitiveCPC, self).__init__()
        self.num_classes = num_classes
        self.proto_dim = proto_dim
        self.temperature = temperature
        self.enable_refinement = enable_refinement
        self.alignment_weight = alignment_weight
        self.gate_diversity_weight = gate_diversity_weight

        # independent branch temperatures: HF and LF differ in discrimination difficulty
        self.hf_temperature = hf_temperature if hf_temperature is not None else temperature
        self.lf_temperature = lf_temperature if lf_temperature is not None else temperature

        # HF class prototypes (K, D) - capture texture-level inter-class differences
        self.hf_prototypes = nn.Parameter(torch.randn(num_classes, proto_dim))
        nn.init.xavier_uniform_(self.hf_prototypes.unsqueeze(0))

        # LF class prototypes (K, D) - capture structural-level inter-class differences
        self.lf_prototypes = nn.Parameter(torch.randn(num_classes, proto_dim))
        nn.init.xavier_uniform_(self.lf_prototypes.unsqueeze(0))

        # HF projection head: HF features -> prototype embedding space
        self.hf_proj = nn.Sequential(
            nn.Conv2d(in_channels, proto_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proto_dim),
            nn.ReLU(inplace=True)
        )

        # LF projection head: LF features -> prototype embedding space
        self.lf_proj = nn.Sequential(
            nn.Conv2d(in_channels, proto_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proto_dim),
            nn.ReLU(inplace=True)
        )

        # HF residual reconstruction
        self.hf_rebuild = nn.Sequential(
            nn.Conv2d(proto_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # LF residual reconstruction
        self.lf_rebuild = nn.Sequential(
            nn.Conv2d(proto_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # frequency gating: learn per pixel whether to rely on HF or LF from the decoder features
        # output (B, 1, H, W), value range [0,1]
        # gate -> 1 means relying more on HF; gate -> 0 means relying more on LF
        # the last Conv2d has explicit bias init to avoid the sigmoid output getting stuck at 0.5
        gate_hidden = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        gate_out = nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        # key init: make the gate output spatially varying rather than a uniform 0.5
        nn.init.zeros_(gate_out.bias)
        nn.init.kaiming_normal_(gate_out.weight, nonlinearity='sigmoid')

        self.freq_gate = nn.Sequential(
            gate_hidden,
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            gate_out,
            nn.Sigmoid()
        )

    def _compute_similarity(self, features, prototypes):
        """
        Compute pixel-prototype cosine similarity
        Args:
            features: (B, D, H, W)
            prototypes: (K, D)
        Returns:
            sim: (B, K, H, W)
        """
        B, D, H, W = features.shape
        q_flat = features.flatten(2)  # (B, D, N)
        q_norm = F.normalize(q_flat, dim=1)  # L2 normalize
        p_norm = F.normalize(prototypes, dim=-1)  # (K, D)

        # (B, D, N) x (D, K) -> (B, N, K) -> (B, K, H, W)
        sim = torch.bmm(q_norm.transpose(1, 2), p_norm.t().unsqueeze(0).expand(B, -1, -1))
        sim = sim.permute(0, 2, 1).view(B, self.num_classes, H, W)
        return sim

    def _compute_refinement(self, attn, prototypes, rebuild_layer):
        """
        Prototype-weighted refinement
        Args:
            attn: (B, K, H, W) softmax affinity
            prototypes: (K, D)
            rebuild_layer: Conv1x1 reconstruction layer
        Returns:
            refinement: (B, C, H, W)
        """
        B, _, H, W = attn.shape
        p_norm = F.normalize(prototypes, dim=-1)  # (K, D)
        attn_flat = attn.flatten(2)  # (B, K, N)

        # (B, D, K) x (B, K, N) -> (B, D, N)
        proto_feat = torch.bmm(
            p_norm.unsqueeze(0).expand(B, -1, -1).transpose(1, 2),
            attn_flat
        )
        proto_feat = proto_feat.view(B, self.proto_dim, H, W)
        return rebuild_layer(proto_feat)

    def forward(self, hf_feat, lf_feat, decoder_feat, gt_labels=None):
        """
        Args:
            hf_feat: HF fused feature (B, C, H_hf, W_hf) - from DWT + BasicBlockH fusion
            lf_feat: LF fused feature (B, C, H_lf, W_lf) - from DWT + BasicBlockL fusion
            decoder_feat: final decoder feature F1 (B, C, H, W)
            gt_labels: ground-truth labels (B, H_gt, W_gt), used during training
        Returns:
            refined: refined feature (B, C, H, W)
            contrastive_loss: contrastive-loss dict {'hf': ..., 'lf': ..., 'total': ...}; None at inference
        """
        B, C, H, W = decoder_feat.shape

        # ===== Step 1: project into each prototype embedding space =====
        q_hf = self.hf_proj(hf_feat)   # (B, D, H_hf, W_hf)
        q_lf = self.lf_proj(lf_feat)   # (B, D, H_lf, W_lf)

        # ===== Step 2: compute pixel-prototype cosine similarity separately =====
        sim_hf = self._compute_similarity(q_hf, self.hf_prototypes)  # (B, K, H_hf, W_hf)
        sim_lf = self._compute_similarity(q_lf, self.lf_prototypes)  # (B, K, H_lf, W_lf)

        # ===== Step 3: compute softmax affinity (for refinement weighting, using a fixed base temperature) =====
        attn_hf = F.softmax(sim_hf / self.temperature, dim=1)  # (B, K, H_hf, W_hf)
        attn_lf = F.softmax(sim_lf / self.temperature, dim=1)  # (B, K, H_lf, W_lf)

        # ===== Step 4: frequency-aware feature refinement =====
        gate = None
        if self.enable_refinement:
            # prototype-weighted refinement (each at its own resolution)
            hf_refine = self._compute_refinement(attn_hf, self.hf_prototypes, self.hf_rebuild)
            lf_refine = self._compute_refinement(attn_lf, self.lf_prototypes, self.lf_rebuild)

            # upsample to the decoder feature resolution
            hf_refine_up = F.interpolate(hf_refine, size=(H, W), mode='bilinear', align_corners=False)
            lf_refine_up = F.interpolate(lf_refine, size=(H, W), mode='bilinear', align_corners=False)

            # frequency gating: learn a spatially adaptive frequency-reliance weight from the decoder features
            gate = self.freq_gate(decoder_feat)  # (B, 1, H, W), in [0,1]

            # gated fusion + residual connection
            refined = decoder_feat + gate * hf_refine_up + (1 - gate) * lf_refine_up
        else:
            refined = decoder_feat

        # ===== Step 5: contrastive loss (training only) =====
        contrastive_loss = None
        if self.training and gt_labels is not None:
            # downsample gt_labels to each frequency feature's resolution
            hf_H, hf_W = sim_hf.shape[2], sim_hf.shape[3]
            lf_H, lf_W = sim_lf.shape[2], sim_lf.shape[3]

            gt_hf = F.interpolate(
                gt_labels.unsqueeze(1).float(), size=(hf_H, hf_W), mode='nearest'
            ).squeeze(1).long()
            gt_lf = F.interpolate(
                gt_labels.unsqueeze(1).float(), size=(lf_H, lf_W), mode='nearest'
            ).squeeze(1).long()

            # HF contrastive loss: InfoNCE (using the HF-specific temperature)
            sim_hf_scaled = sim_hf.permute(0, 2, 3, 1).reshape(-1, self.num_classes) / self.hf_temperature
            labels_hf = gt_hf.flatten()
            hf_loss = F.cross_entropy(sim_hf_scaled, labels_hf)

            # LF contrastive loss: InfoNCE (using the LF-specific temperature)
            sim_lf_scaled = sim_lf.permute(0, 2, 3, 1).reshape(-1, self.num_classes) / self.lf_temperature
            labels_lf = gt_lf.flatten()
            lf_loss = F.cross_entropy(sim_lf_scaled, labels_lf)

            total_loss = (hf_loss + lf_loss) / 2.0

            # --- auxiliary loss A: cross-frequency prototype alignment loss ---
            # prototypes of the same class in HF and LF should keep a positive similarity (complementary);
            # rather than fully orthogonal (current diagnosis shows cross-frequency diagonal similarity = -0.02)
            hf_p_norm = F.normalize(self.hf_prototypes, dim=-1)  # (K, D)
            lf_p_norm = F.normalize(self.lf_prototypes, dim=-1)  # (K, D)
            # diagonal elements: cross-frequency similarity of the same class
            cross_freq_sim = torch.sum(hf_p_norm * lf_p_norm, dim=-1)  # (K,)
            # target: cross-frequency same-class similarity > 0.3 (complementary rather than orthogonal)
            alignment_loss = F.relu(0.3 - cross_freq_sim).mean()

            # --- auxiliary loss B: gated spatial-diversity regularization ---
            # prevent the gate from collapsing to a uniform 0.5: encourage high spatial variance in the gate
            gate_diversity_loss = torch.tensor(0.0, device=decoder_feat.device)
            if gate is not None:
                # compute the spatial std of the gate within each sample, averaged over the batch
                gate_std = gate.flatten(1).std(dim=1).mean()  # scalar
                # larger variance is better, so negate it as the loss (minimizing negative variance = maximizing variance)
                gate_diversity_loss = -gate_std

            total_loss = total_loss + self.alignment_weight * alignment_loss \
                                + self.gate_diversity_weight * gate_diversity_loss

            contrastive_loss = {
                'hf': hf_loss,
                'lf': lf_loss,
                'alignment': alignment_loss,
                'gate_diversity': gate_diversity_loss,
                'total': total_loss
            }

        return refined, contrastive_loss


class ConfusionBoundaryCPC(nn.Module):
    """
    Confusion-aware boundary-enhanced prototype contrast module (Confusion-aware & Contrast-enhanced CPC, CC-CPC)

    On top of the standard CPC, two novelties are introduced:
    1. confusion-aware weighting (Confusion-Aware Weighting):
       - dynamically track the inter-class confusion frequency
       - apply a stronger contrastive push to highly confused class pairs
       - explicitly add a prototype-margin constraint to push the prototypes of confused classes apart

    2. boundary-enhanced contrast (Boundary-Enhanced Contrast):
       - use the edge GT to identify the target boundary regions
       - use a lower temperature at boundaries (sharper contrast) to improve discrimination
       - use a normal temperature in interior regions to keep the features smooth

    Design motivation:
    - Tank <-> Military Vehicle are frequently confused due to similar camouflage paint
    - the core challenge of camouflaged targets is the boundary - target and background blend gradually
    - standard CPC treats all pixels and class pairs equally, lacking specificity
    """

    def __init__(self, in_channels=128, num_classes=5, proto_dim=64,
                 temperature=0.07, enable_refinement=True,
                 confusion_momentum=0.7, boundary_sharpness=0.85,
                 separation_margin=0.9):
        super(ConfusionBoundaryCPC, self).__init__()
        self.num_classes = num_classes
        self.proto_dim = proto_dim
        self.temperature = temperature
        self.enable_refinement = enable_refinement
        self.confusion_momentum = confusion_momentum
        self.boundary_sharpness = boundary_sharpness
        self.separation_margin = separation_margin

        # ===== standard CPC components =====
        self.prototypes = nn.Parameter(torch.randn(num_classes, proto_dim))
        nn.init.xavier_uniform_(self.prototypes.unsqueeze(0))

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, proto_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(proto_dim),
            nn.ReLU(inplace=True)
        )

        self.rebuild = nn.Sequential(
            nn.Conv2d(proto_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # ===== confusion-aware components =====
        # runtime confusion matrix (EMA-updated, not part of the gradient)
        self.register_buffer('confusion_matrix', torch.zeros(num_classes, num_classes))
        self.register_buffer('class_confusion_score', torch.ones(num_classes))
        self.register_buffer('confusion_initialized', torch.tensor(False))

    def update_confusion_matrix(self, pred_labels, gt_labels):
        """
        EMA-update the confusion matrix
        Args:
            pred_labels: model prediction (B, H, W), from argmax of sim
            gt_labels: ground-truth labels (B, H, W)
        """
        with torch.no_grad():
            batch_cm = torch.zeros(self.num_classes, self.num_classes,
                                   device=gt_labels.device)
            gt_flat = gt_labels.flatten()
            pred_flat = pred_labels.flatten()

            # count foreground classes only (ignore the large pixel deviation background may cause)
            for c_gt in range(self.num_classes):
                mask = (gt_flat == c_gt)
                if mask.sum() == 0:
                    continue
                pred_for_class = pred_flat[mask]
                for c_pred in range(self.num_classes):
                    batch_cm[c_gt, c_pred] = (pred_for_class == c_pred).float().sum()

            # row normalization
            row_sum = batch_cm.sum(dim=1, keepdim=True)
            row_sum = torch.clamp(row_sum, min=1.0)
            batch_cm = batch_cm / row_sum

            # EMA update
            if not self.confusion_initialized:
                self.confusion_matrix.copy_(batch_cm)
                self.confusion_initialized.fill_(True)
            else:
                self.confusion_matrix.mul_(self.confusion_momentum).add_(
                    batch_cm * (1 - self.confusion_momentum)
                )

            # compute the per-class confusion score (the fraction misclassified)
            for c in range(self.num_classes):
                self.class_confusion_score[c] = 1.0 - self.confusion_matrix[c, c]

    def _compute_confusion_weights(self, gt_labels):
        """
        Compute the confusion-aware weight per pixel
        High-confusion classes -> higher weights (the model should pay more attention to these classes)
        Returns:
            weights: (B*N,) weight per pixel
        """
        # class_confusion_score: (K,), value range [0, 1]
        # map to [1.0, 2.0] so the minimum weight is 1
        weights = 1.0 + self.class_confusion_score  # (K,)
        labels_flat = gt_labels.flatten()  # (B*N,)
        pixel_weights = weights[labels_flat]  # (B*N,)
        return pixel_weights

    def _compute_separation_loss(self):
        """
        Prototype-margin loss: explicitly push the prototypes of confused class pairs apart
        For class pairs with high confusion scores, apply a stronger separation constraint
        Ensure the loss activates early in training via a minimum confusion weight
        Returns:
            sep_loss: scalar
        """
        p_norm = F.normalize(self.prototypes, dim=-1)  # (K, D)

        # compute the prototype similarity for all class pairs
        sim_matrix = torch.mm(p_norm, p_norm.t())  # (K, K)

        # take the upper triangle of the confusion matrix (excluding the diagonal) as weights
        # add a minimum weight so the separation loss activates early in training
        min_confusion_weight = 0.1
        confusion_weights = torch.zeros_like(sim_matrix)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:
                    # take the max confusion score across both directions, no less than the minimum weight
                    confusion_weights[i, j] = max(
                        self.confusion_matrix[i, j].item(),
                        self.confusion_matrix[j, i].item(),
                        min_confusion_weight
                    )

        # separation loss: for highly confused class pairs, penalize high prototype similarity
        # we want similarity < margin; the part exceeding margin is penalized
        sep_loss = confusion_weights * F.relu(sim_matrix - self.separation_margin)
        # exclude the diagonal and take the mean
        mask = (~torch.eye(self.num_classes, dtype=torch.bool, device=sim_matrix.device))
        sep_loss = sep_loss[mask].mean()

        return sep_loss

    def _compute_boundary_temperature(self, edge_map, H, W):
        """
        Boundary-adaptive temperature
        Lower temperature at boundaries (sharper contrast), normal temperature inside
        Args:
            edge_map: edge GT (B, 1, H_edge, W_edge) or (B, H_edge, W_edge)
            H, W: feature-map resolution
        Returns:
            temp_map: (B, 1, H, W), per-pixel temperature
        """
        # ensure 4D: (B, 1, H, W)
        if edge_map.dim() == 3:
            edge_map = edge_map.unsqueeze(1)

        # downsample to the feature-map resolution
        edge_resized = F.interpolate(
            edge_map.float(),
            size=(H, W), mode='bilinear', align_corners=False
        )  # (B, 1, H, W)

        # at boundaries the temperature drops: tau_boundary = tau * (1 - boundary_sharpness * edge)
        temp_map = self.temperature * (1.0 - self.boundary_sharpness * edge_resized)

        # temperature lower-bound clamp: no less than 1/10 of the base temperature to avoid numerical instability
        temp_map = torch.clamp(temp_map, min=self.temperature * 0.1)

        return temp_map

    def forward(self, x, gt_labels=None, edge_map=None):
        """
        Args:
            x: decoder feature (B, C, H, W)
            gt_labels: ground-truth labels (B, H_gt, W_gt), used during training
            edge_map: edge GT (B, H_edge, W_edge), used during training
        Returns:
            refined: refined feature (B, C, H, W)
            loss_dict: {'contrastive': ..., 'separation': ..., 'total': ...}
        """
        B, C, H, W = x.shape

        # downsample GT to the feature-map size
        if gt_labels is not None:
            gt_H, gt_W = gt_labels.shape[1], gt_labels.shape[2]
            if gt_H != H or gt_W != W:
                gt_labels = F.interpolate(
                    gt_labels.unsqueeze(1).float(), size=(H, W),
                    mode='nearest'
                ).squeeze(1).long()

        # Step 1: project into the prototype space
        q = self.proj(x)  # (B, D, H, W)

        # Step 2: pixel-prototype cosine similarity
        q_flat = q.flatten(2)  # (B, D, N)
        q_norm = F.normalize(q_flat, dim=1)
        p_norm = F.normalize(self.prototypes, dim=-1)  # (K, D)

        sim = torch.bmm(q_norm.transpose(1, 2), p_norm.t().unsqueeze(0).expand(B, -1, -1))
        sim = sim.permute(0, 2, 1).view(B, self.num_classes, H, W)  # (B, K, H, W)

        # Step 3: softmax affinity (for refinement, using a fixed temperature for stability)
        attn = F.softmax(sim / self.temperature, dim=1)  # (B, K, H, W)

        # prototype-weighted refinement
        attn_flat = attn.flatten(2)  # (B, K, N)
        proto_feat = torch.bmm(
            p_norm.unsqueeze(0).expand(B, -1, -1).transpose(1, 2), attn_flat
        )
        proto_feat = proto_feat.view(B, self.proto_dim, H, W)

        if self.enable_refinement:
            refined = x + self.rebuild(proto_feat)
        else:
            refined = x

        # Step 4: loss computation (training only)
        loss_dict = None
        if self.training and gt_labels is not None:
            # --- 4a: update the confusion matrix ---
            pred_labels = sim.detach().argmax(dim=1)  # (B, H, W)
            self.update_confusion_matrix(pred_labels, gt_labels)

            # --- 4b: confusion-aware weighted contrastive loss ---
            if edge_map is not None:
                temp_map = self._compute_boundary_temperature(edge_map, H, W)
                # per-pixel temperature: (B, 1, H, W)
                sim_scaled = sim / temp_map  # broadcast: (B, K, H, W) / (B, 1, H, W)
            else:
                sim_scaled = sim / self.temperature

            sim_scaled = sim_scaled.permute(0, 2, 3, 1).reshape(B * H * W, self.num_classes)
            labels = gt_labels.flatten()

            # base contrastive loss
            contrastive_loss = F.cross_entropy(sim_scaled, labels, reduction='none')

            # confusion-aware weighting
            confusion_weights = self._compute_confusion_weights(gt_labels)
            contrastive_loss = (confusion_weights * contrastive_loss).mean()

            # --- 4c: prototype-distance loss ---
            separation_loss = self._compute_separation_loss()

            loss_dict = {
                'contrastive': contrastive_loss,
                'separation': separation_loss,
                'total': contrastive_loss + 0.5 * separation_loss
            }

        return refined, loss_dict
















