# ------------------------------------------------------------------
# 2D CNN Encoder
# Contact person: Mohamad Hakam, Shams Eddin <shams@iai.uni-bonn.de>
# ------------------------------------------------------------------

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from models.channel_attention import ChannelAttention_2D

# ------------------------------------------------------------------

class conv_block(nn.Module):
    """ Residual CNN Block """
    def __init__(self, in_channels=96, out_channels=96):
        super(conv_block, self).__init__()

        """
        Parameters
        ----------
        in_channels : int (default 96)
            number of input channels
        out_channels : int (default 96)
            number of output channels
        """

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.norm_layer1 = nn.LayerNorm(self.out_channels)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.norm_layer2 = nn.LayerNorm(self.out_channels)

        self.conv_shortcut = nn.Conv2d(self.in_channels, self.out_channels,
                                       kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.norm_shortcut = nn.LayerNorm(self.out_channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x : input torch.tensor [N, C, H, W]
        """
        x_short = self.conv_shortcut(x)

        N, C, W, H = x_short.shape
        x_short = self.norm_shortcut(x_short.permute(0, 2, 3, 1).reshape(N * W * H, C))
        x_short = x_short.view(N, H, W, C).permute(0, 3, 2, 1)
        x_short = self.act(x_short)

        x = self.conv1(x)
        N, C, W, H = x.shape
        x = self.norm_layer1(x.permute(0, 2, 3, 1).reshape(N * W * H, C))
        x = x.view(N, H, W, C).permute(0, 3, 1, 2)
        x = self.act(x)

        x = self.conv2(x)
        N, C, W, H = x.shape
        x = self.norm_layer2(x.permute(0, 2, 3, 1).reshape(N * W * H, C))
        x = x.view(N, H, W, C).permute(0, 3, 1, 2)
        x = self.act(x)

        return x + x_short


class CNN_2D(nn.Module):
    """ 2D CNN Encoder """
    def __init__(self, config):
        super(CNN_2D, self).__init__()

        """
        Parameters
        ----------
        config : argparse
           configuration file from config.py
        """

        # read configuration from file
        self.out_channels = config.out_channels
        self.n_layers = len(config.out_channels)

        self.in_channels = [config.in_channels]

        for i in range(self.n_layers - 1):
            self.in_channels.append(self.out_channels[i])

        self.down_sampling = config.en_down_sampling

        # create conv layers with two blocks
        self.conv_blocks = nn.ModuleList([])
        for layer in range(self.n_layers):
            self.conv_blocks.append(conv_block(self.in_channels[layer], self.out_channels[layer]))

        # create down sampling layers
        self.down_sampling_block = nn.ModuleList([])

        for layer in range(self.n_layers - 1):

            if self.down_sampling == 'conv':

                block = [nn.Conv2d(self.out_channels[layer], self.out_channels[layer],
                                   kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=True),
                         nn.ReLU(inplace=True)]
                self.down_sampling_block.append(nn.Sequential(*block))
            elif self.down_sampling == 'max':
                block = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
                self.down_sampling_block.append(block)
            else:
                raise ValueError('%s is not a recognized down_sampling type. Supported are conv or max' % self.down_sampling)

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

        for m in self.conv_blocks.children():
            if hasattr(m, '_init_weights'):
                m._init_weights()

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

        for i in range(self.n_layers):

            x = self.conv_blocks[i](x)

            x_out.append(x)

            if i != self.n_layers - 1:
                x = self.down_sampling_block[i](x)

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

    test_x = torch.randn((2, 65+6, 397, 409), device=device)

    model = CNN_2D(config_file).to(device)
    print(model)

    test = model(test_x)
    for test_x in test:
        print(test_x.shape)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

