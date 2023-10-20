# ------------------------------------------------------------
# Channel Attention
# Contact person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>

# Based on https://github.com/Jongchan/attention-module/blob/master/MODELS/model_resnet.py
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------

class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.view(x.size(0), -1)


class ChannelAttention_2D(nn.Module):
    """ Channel Attention in 2D """
    def __init__(self, channels_in: int = 65, reduction_ratio: int = 5, pooling_type: list = ['mean', 'std']):
        super(ChannelAttention_2D, self).__init__()

        """
        Parameters
        ----------
        in_channels : int (default 65)
            number of input channels
        reduction_ratio : int (default 5)
            reduction ratio on the input channels
        pooling_type : list (default ['mean', 'std'])
            type of pooling to be used for the attention
        """

        self.channels_in = channels_in
        self.reduction_ratio = reduction_ratio
        self.pooling_type = pooling_type

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(channels_in, channels_in // reduction_ratio),
            nn.GELU(),
            nn.Linear(channels_in // reduction_ratio, channels_in)
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : input torch.tensor [N, C, H, W]
        """

        channel_att_sum = None

        for pool_type in self.pooling_type:
            if pool_type == 'mean':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'std':
                std_pool = torch.std(x, dim=(-1, -2), keepdim=True)
                channel_att_raw = self.mlp(std_pool)
            #elif pool_type == 'lp':
            #    lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
            #    channel_att_raw = self.mlp(lp_pool)
            #elif pool_type == 'lse':
            #    # LSE pool only
            #    lse_pool = logsumexp_2d(x)
            #    channel_att_raw = self.mlp(lse_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scaled_x = torch.sigmoid(channel_att_sum)

        return scaled_x



if __name__ == '__main__':

    device = 'cpu'

    test = torch.randn([2, 65, 128, 128]).to(device)
    model = ChannelAttention_2D(channels_in=65, reduction_ratio=5, pooling_type=['max', 'mean', 'std']).to(device)
    test_scaled = model(test)

    print(test_scaled.shape)
    print(test_scaled[0, :10])

    test = test * test_scaled[:, :, None, None]
    print(test.shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

