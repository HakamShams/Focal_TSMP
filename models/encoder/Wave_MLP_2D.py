# ------------------------------------------------------------
# 2D Wave-MLP Encoder
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
from models.channel_attention import ChannelAttention_2D

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

    def __init__(self, dim, input_resolution, mlp_ratio=4., qkv_bias=False, drop=0.,
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
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, out_dim, input_resolution, depth,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 downsample=None, use_checkpoint=False,
                 mode='fc', qkv_bias=False,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            WaveBlock(
                dim=dim,
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=nn.GELU,
                mode=mode
            )

            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution,
                patch_size=2,
                in_chans=out_dim,
                embed_dim=dim,
                norm_layer=None,
            )
        else:
            self.downsample = None

    def forward(self, x, H, W):

        if self.downsample is not None:
            x = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)
            x, Ho, Wo = self.downsample(x)
        else:
            Ho, Wo = H, W

        for blk in self.blocks:
            blk.H, blk.W = Ho, Wo
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        return x, Ho, Wo


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size: tuple = (224, 224), patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):

        x = self.proj(x)
        H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, H, W



class Wave_MLP_2D(nn.Module):
    """ 2D Wave-MLP Encoder """

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
        self.out_channels = config.out_channels
        self.n_layers = len(config.out_channels)
        self.depths = config.en_depths
        self.patch_size = config.en_patch_size
        self.in_channels = config.out_channels
        self.in_chans = config.in_channels
        self.embed_dim = config.out_channels[0]
        self.mlp_ratio = config.en_mlp_ratio
        self.ape = config.en_ape
        self.drop_rate = config.en_drop_rate
        self.drop_path_rate = config.en_drop_path_rate
        self.patch_norm = config.en_patch_norm
        self.use_checkpoint = config.en_use_checkpoint

        self.pretrained = config.en_pretrained if config.phase == 'train' else None

        self.qkv_bias = config.en_qkv_bias
        self.mode = 'fc'

        # split image into patches using either non-overlapped embedding or overlapped embedding
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=(self.n_lat, self.n_lon), patch_size=self.patch_size,
                                      in_chans=self.in_chans, embed_dim=self.embed_dim,
                                      norm_layer=nn.LayerNorm if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        self.input_resolution = []
        for i in range(self.n_layers):
            self.input_resolution.append((patches_resolution[0] // (2 ** i), patches_resolution[1] // (2 ** i)))

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.n_layers):
            layer = BasicLayer(dim=self.in_channels[i_layer],
                               out_dim=self.in_channels[i_layer - 1] if (i_layer < self.n_layers) else None,
                               input_resolution=self.input_resolution[i_layer],
                               depth=self.depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=self.drop_rate,
                               drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                               downsample=PatchEmbed if (i_layer != 0) else None,
                               use_checkpoint=self.use_checkpoint,
                               mode=self.mode,
                               qkv_bias=self.qkv_bias,
                               )

            self.layers.append(layer)

        self.en_channel_att = config.en_channel_att

        # create channel attention layers
        if self.en_channel_att:
            self.en_channel_att_r = config.en_channel_att_r
            self.en_channel_att_p = config.en_channel_att_p
            self.channel_attn = ChannelAttention_2D(channels_in=self.in_chans,
                                                    reduction_ratio=self.en_channel_att_r,
                                                    pooling_type=self.en_channel_att_p)

        self._init_weights()

    def _init_weights(self):
        def init_func(m):

            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(init_func)

        if self.pretrained:
            print('initialize weights from pretrained model {} ...'.format(self.pretrained))
            state_dict = torch.load(self.pretrained, map_location='cpu')

            # delete head and last norm layer
            del state_dict['head.weight'], state_dict['head.bias'], state_dict['norm.weight'], state_dict['norm.bias']

            networks_index_keys = sorted([k for k in state_dict.keys() if "network" in k])
            layers_index_keys = sorted([k.replace('network', 'layers') for k in state_dict.keys() if "network" in k])

            layers_index_keys = [k[:8] + '.blocks' + k[8:] for k in layers_index_keys]

            for k, j in zip(layers_index_keys, networks_index_keys):
                state_dict[k] = state_dict[j]
            for k in networks_index_keys:
                del state_dict[k]

            layers_index_keys = sorted([k for k in state_dict.keys() if "layers" in k])

            layers_index_keys_tmp = []
            for k in layers_index_keys:
                if k[7] == 't':
                    layers_index_keys_tmp.append(k)
                    continue
                n = int(k[7])

                if n == 2:
                    k = k[:7] + '1' + k[8:]
                elif n == 4:
                    k = k[:7] + '2' + k[8:]
                elif n == 6:
                    k = k[:7] + '3' + k[8:]

                layers_index_keys_tmp.append(k)

            for k, j in zip(layers_index_keys_tmp, layers_index_keys):
                state_dict[k] = state_dict[j]

            # prepare patch embed weights
            m1 = state_dict['patch_embed.proj.weight'].mean()
            s1 = state_dict['patch_embed.proj.weight'].std()
            state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].mean(axis=(1, 2, 3)). \
                unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, self.in_chans, self.patch_size, self.patch_size)
            m2 = state_dict['patch_embed.proj.weight'].mean()
            s2 = state_dict['patch_embed.proj.weight'].std()
            state_dict['patch_embed.proj.weight'] = m1 + (state_dict['patch_embed.proj.weight'] - m2) * s1 / s2

            self.load_state_dict(state_dict, strict=False)
            del state_dict
            torch.cuda.empty_cache()


    def forward(self, x):
        """
        Args:
            x : input torch.tensor [N, C, H, W]
        """

        x_out = []

        N, C, H, W = x.shape

        if self.en_channel_att:
            x_scaled = self.channel_attn(x[:, :, :, :])
            x[:, :, :, :] = x[:, :, :, :] * x_scaled[:, :, None, None]

        x, H, W = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed

        for i, layer in enumerate(self.layers):
            x, H, W = layer(x, H, W)
            h, w, c = self.input_resolution[i][0], self.input_resolution[i][1], self.out_channels[i]

            x_out.append(x.view(N, h, w, c).permute(0, 3, 1, 2))

        return x_out


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

    test_x = torch.randn((2, 65 + 6, 397, 409), device=device)

    model = Wave_MLP_2D(config_file).to(device)
    print(model)

    test = model(test_x)
    for test_x in test:
        print(test_x.shape)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

