# ------------------------------------------------------------
# Regression Head
# Contact person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
# ------------------------------------------------------------

import torch
import torch.nn as nn

# ------------------------------------------------------------

class regression_head(nn.Module):
    """ Regression Head """
    def __init__(self, in_channels=288):
        super(regression_head, self).__init__()

        """
        Parameters
        ----------
        in_channels : int (default 288)
            number of input channels
        """

        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels // 2, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(self.in_channels // 2, 1, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1), bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """
        x : torch.tensor [N, C, H, W]
        """
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv3(x)

        return x

