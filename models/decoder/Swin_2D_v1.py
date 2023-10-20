# ------------------------------------------------------------
# 2D Swin Transformer V1 Encoder
# Contact person: Mohamad Hakam, Shams Eddin <shams@iai.uni-bonn.de>

# The original code is available at https://github.com/microsoft/Swin-Transformer
# and is licensed under The MIT License. Copyright (c) 2021 Microsoft
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.regression_head import regression_head

import numpy as np

try:
    import os, sys
    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from models.kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

# ------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, out_dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution

            # calculate attention mask for SW-MSA
            Hp = int(np.ceil(H / self.window_size)) * self.window_size
            Wp = int(np.ceil(W / self.window_size)) * self.window_size

            img_mask = torch.zeros((1, Hp, Wp, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0

            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"


        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, Hp, Wp, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, Hp, Wp, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): Upsampling layer at the end of the layer. Default: None
        upsampling_type (str): Upsampling type. Default: bilinear
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., upsample=None, upsampling_type='bilinear', qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.upsampling_type = upsampling_type

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, out_dim=None, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(
                upsampling=upsampling_type,
                patch_size=2,
                in_chans=dim,
                embed_dim=out_dim,
            )
        else:
            self.upsample = None

    def forward(self, x, h=None, w=None):

        H, W = x.shape[-2:]

        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)

        if self.upsample is not None:
            x, Ho, Wo = self.upsample(x)

        return x


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


class Swin_2D_v1(nn.Module):
    """ 2D Swin Transformer V1 Decoder """
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
        self.depths = config.de_depths
        self.patch_size = config.en_patch_size
        self.in_channels = config.out_channels

        self.in_chans = config.in_channels
        self.num_heads = config.de_n_heads
        self.window_size = config.de_window_size
        self.mlp_ratio = config.de_mlp_ratio
        self.drop_rate = config.de_drop_rate
        self.attn_drop_rate = config.de_attn_drop_rate
        self.drop_path_rate = config.de_drop_path_rate
        self.qkv_bias = config.de_qkv_bias
        self.qk_scale = config.de_qk_scale

        self.use_checkpoint = config.de_use_checkpoint
        self.fused_window_process = config.de_fused_window_process

        self.up_sampling = config.de_up_sampling

        patches_resolution = [self.n_lat // self.patch_size, self.n_lon // self.patch_size]
        self.patches_resolution = patches_resolution

        self.input_resolution = []
        for i in reversed(range(self.n_layers)):
            if i != 0:
                hh = int(np.ceil(patches_resolution[0] / (2 ** i)))
                ww = int(np.ceil(patches_resolution[1] / (2 ** i)))
            else:
                hh = patches_resolution[0]
                ww = patches_resolution[1]
            self.input_resolution.append((hh, ww))

        self.out_channels = []
        self.in_channels = [config.out_channels[-1]]

        for i in reversed(range(self.n_layers - 1)):
            if i == (self.n_layers - 2):
                self.out_channels.append(config.out_channels[i + 1] // 2)
            else:
                self.out_channels.append(config.out_channels[i + 1])
            self.in_channels.append(config.out_channels[i] + self.out_channels[(self.n_layers - 2) - i])

        self.out_channels.append(config.out_channels[0] * 3)

        # stochastic depth
        dpr = [x.item() for x in reversed(torch.linspace(0, self.drop_path_rate, sum(self.depths)))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()

        for i_layer in range(self.n_layers):
            layer = BasicLayer(dim=self.in_channels[i_layer],
                               out_dim=self.out_channels[i_layer],
                               input_resolution=self.input_resolution[i_layer],
                               depth=self.depths[i_layer],
                               num_heads=self.num_heads[i_layer],
                               window_size=self.window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                               drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                               drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,
                               upsample=PatchEmbed if (i_layer != (self.n_layers - 1)) else None,
                               upsampling_type=self.up_sampling,
                               use_checkpoint=self.use_checkpoint,
                               fused_window_process=self.fused_window_process)

            self.layers.append(layer)

        # create regression head
        self.BT_block = regression_head(self.out_channels[-1])
        self.NDVI_block = regression_head(self.out_channels[-1])

        self._init_weights()

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
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
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

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
            x_out = self.layers[i](x_i, H, W)


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

    test_x = [torch.randn((1, 96, 397, 409)).to(device), torch.randn((1, 192, 199, 205)).to(device),
              torch.randn((1, 384, 100, 103)).to(device)]

    model = Swin_2D_v1(config_file).to(device)
    print(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    BT, NDVI = model(test_x)

    print(BT.shape)
    print(NDVI.shape)



