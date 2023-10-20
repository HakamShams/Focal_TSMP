# ------------------------------------------------------------
# 2D Wave-MLP Decoder
# Contact person: Mohamad Hakam, Shams Eddin <shams@iai.uni-bonn.de>

# The original code is available at https://gitee.com/mindspore/models/blob/master/research/cv/wave_mlp/
# and https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/wavemlp_pytorch
# and is licensed under the Apache License. Copyright 2022 Huawei Technologies Co., Ltd
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
import torch.utils.checkpoint as checkpoint
from models.regression_head import regression_head

# ------------------------------------------------------------


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, proj_drop=0., mode='fc'):
        super().__init__()

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_c = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)

        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), stride=1, padding=(0, 7 // 2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), stride=1, padding=(7 // 2, 0), groups=dim, bias=False)

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode

        if mode == 'fc':
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
        else:
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x):

        B, C, H, W = x.shape

        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)

        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)

        #         x_1=self.fc_h(x)
        #         x_2=self.fc_w(x)
        #         x_h=torch.cat([x_1*torch.cos(theta_h),x_2*torch.sin(theta_h)],dim=1)
        #         x_w=torch.cat([x_1*torch.cos(theta_w),x_2*torch.sin(theta_w)],dim=1)

        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WaveBlock(nn.Module):

    def __init__(self, dim, out_dim, input_resolution, mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.GroupNorm, mode='fc'):
        super().__init__()

        self.input_resolution = input_resolution


        self.H = input_resolution[0]
        self.W = input_resolution[1]

        self.norm1 = norm_layer(1, dim)
        self.attn = PATM(dim, qkv_bias=qkv_bias, mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(1, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):

        B, L, C = x.shape
        x = x.view(-1, self.H, self.W, C).permute(0, 3, 1, 2)

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 2, 3, 1).view(B, L, C)

        return x


class BasicLayer(nn.Module):
    """ A basic Transformer layer for one stage.

       Args:
           dim (int): Number of input channels.
           input_resolution (tuple[int]): Input resolution.
           depth (int): Number of blocks.
           window_size (int): Local window size.
           mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
           qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
           attn_drop_rate (float): Attention dropout rate. Default: 0
           drop (float, optional): Dropout rate. Default: 0.0
           drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
           upsample (nn.Module | None, optional): Upsampling layer at the end of the layer. Default: None
           upsampling_type (str): Upsampling type. Default: bilinear
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
       """

    def __init__(self, dim, out_dim, input_resolution, depth,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 upsample=None, upsampling_type='bilinear', use_checkpoint=False,
                 mode='fc', qkv_bias=False
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.upsampling_type = upsampling_type

        # build blocks
        self.blocks = nn.ModuleList([
            WaveBlock(
                dim=dim,
                out_dim=None,
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=nn.GELU,
                mode=mode
            )

            for i in range(depth)])

        if upsample is not None:
            self.upsample = upsample(
                upsampling=upsampling_type,
                patch_size=2,
                in_chans=dim,
                embed_dim=out_dim,
            )
        else:
            self.upsample = None

    def forward(self, x, H, W):

        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C

        Ho, Wo = H, W

        for blk in self.blocks:
            blk.H, blk.W = Ho, Wo
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)

        if self.upsample is not None:
            x, Ho, Wo = self.upsample(x)
        else:
            Ho, Wo = H, W

        return x, Ho, Wo


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        upsampling (str): type of the up sampling conv or bilinear.  Default: bilinear.
        patch_size (int): Patch token size. Default: 2.
        in_chans (int): Number of input image channels. Default: 96.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, upsampling='bilinear', patch_size=2, in_chans=96, embed_dim=96):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.up_sampling = upsampling

        if self.up_sampling == 'conv':
            self.proj = nn.Sequential(nn.ConvTranspose2d(self.in_chans, self.embed_dim,
                                                         kernel_size=(patch_size, patch_size),
                                                         stride=(patch_size, patch_size), padding=(0, 0), bias=True))
                                      #nn.ReLU(inplace=True))

        elif self.up_sampling in ('bilinear', 'nearest', 'bicubic'):

            self.proj = nn.Sequential(nn.Upsample(scale_factor=patch_size, mode=self.up_sampling),
                                      nn.Conv2d(self.in_chans,  self.embed_dim, kernel_size=(1, 1),
                                                stride=(1, 1), padding=(0, 0), bias=True))
                                      #nn.ReLU(inplace=True))
        else:
            raise ValueError('%s is not a recognized down_sampling type.\
             Supported are conv, bilinear, nearest or bicubic' % self.up_sampling)

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2:]
        return x, H, W

class Wave_MLP_2D(nn.Module):
    """ 2D Wave-MLP Decoder """

    def __init__(self, config):
        super().__init__()

        """
        Parameters
        ----------
        config : argparse
            configuration file from config.py
        """

        self.n_lat = config.n_lat - (config.cut_lateral_boundary * 2)
        self.n_lon = config.n_lon - (config.cut_lateral_boundary * 2)
        self.depths = config.de_depths
        self.mlp_ratio = config.de_mlp_ratio
        self.drop_rate = config.de_drop_rate
        self.drop_path_rate = config.de_drop_path_rate
        self.use_checkpoint = config.de_use_checkpoint

        self.patch_size = config.en_patch_size
        self.up_sampling = config.de_up_sampling
        self.n_layers = len(config.out_channels)

        self.out_channels = []
        self.in_channels = [config.out_channels[-1]]

        for i in reversed(range(self.n_layers - 1)):
            if i == (self.n_layers - 2):
                self.out_channels.append(config.out_channels[i + 1] // 2)
            else:
                self.out_channels.append(config.out_channels[i + 1])
            self.in_channels.append(config.out_channels[i] + self.out_channels[(self.n_layers - 2) - i])
        self.out_channels.append(config.out_channels[0] * 3)

        patches_resolution = [self.n_lat // self.patch_size, self.n_lon // self.patch_size]
        self.patches_resolution = patches_resolution

        self.input_resolution = []
        for i in reversed(range(self.n_layers)):
            self.input_resolution.append((patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i)))

        # stochastic depth
        dpr = [x.item() for x in
               reversed(torch.linspace(0, self.drop_path_rate, sum(self.depths)))]  # stochastic depth decay rule

        self.qkv_bias = config.en_qkv_bias
        self.mode = 'fc'

        # build layers
        self.layers = nn.ModuleList()

        for i_layer in range(self.n_layers):
            layer = BasicLayer(dim=self.in_channels[i_layer],
                               out_dim=self.out_channels[i_layer],
                               input_resolution=self.input_resolution[i_layer],
                               depth=self.depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=self.drop_rate,
                               drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                               # norm_layer=norm_layer,
                               upsample=PatchEmbed if (i_layer != (self.n_layers - 1)) else None,
                               upsampling_type=self.up_sampling,
                               use_checkpoint=self.use_checkpoint,
                               mode=self.mode,
                               qkv_bias=self.qkv_bias,
                               )

            self.layers.append(layer)

        # create regression head
        self.BT_block = regression_head(self.out_channels[-1])
        self.NDVI_block = regression_head(self.out_channels[-1])

        self._init_weights()

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(init_func)

        for m in self.BT_block.children():
            if hasattr(m, '_init_weights'):
                m._init_weights()
            else:
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    trunc_normal_(m.weight, std=.02)

        for m in self.NDVI_block.children():
            if hasattr(m, '_init_weights'):
                m._init_weights()
            else:
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    trunc_normal_(m.weight, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {''}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {''}

    def forward(self, x):
        """
        Args:
            x : list of input torch.tensor [N, C, H, W]
        """

        _, _, H, W = x[-1].shape

        for i, x_i in enumerate(reversed(x)):
            if i != 0:

                if x_i.size()[-2:] != x_out.size()[-2:]:
                    x_out = F.interpolate(x_out, size=x_i.size()[-2:], mode='bilinear', align_corners=False)

                x_i = torch.cat((x_i, x_out), dim=1)

            _, _, H, W = x_i.shape

            x_out, H, W = self.layers[i](x_i, H, W)

        if self.patch_size != 1:
            #x_out = F.interpolate(x_out, size=(self.n_lat, self.n_lon), mode='bicubic')
            x_out = F.interpolate(x_out, size=(self.n_lat, self.n_lon), mode='bilinear', align_corners=False)

        BT_out = self.BT_block(x_out)[:, 0, :, :]
        NDVI_out = self.NDVI_block(x_out)[:, 0, :, :]

        return BT_out, NDVI_out



if __name__ == '__main__':

    import config
    import os

    config_file = config.read_arguments(train=True, print=False, save=False)

    if config_file.gpu_id != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config_file.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    test_x = [torch.randn((1, 96, 198, 204)).to(device), torch.randn((1, 192, 99, 102)).to(device),
              torch.randn((1, 384, 50, 51)).to(device)]

    model = Wave_MLP_2D(config_file).to(device)
    print(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    BT, NDVI = model(test_x)

    print(BT.shape)
    print(NDVI.shape)



