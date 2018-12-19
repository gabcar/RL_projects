import torch.nn as nn

class AlexNet(nn.Module):
    """
    AlexNet model from the AlexNet Paper

    Comments are direct quotes from the paper, used as
    guidelines when designing the net.
    """

    def __init__(self, n_classes):
        """
        in: 
            n_classes [int] - Number of classes to predict
        out n/a
        """
        super(AlexNet, self).__init__()
        # The first convolutional layer filters the 224x224x3 
        # input image with 96 kernels of size 11×11×3
        # with a stride of 4 pixels
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=96,
                kernel_size=11,
                stride=4,
                padding=0
            ),
            nn.ReLU()
        )
        # Layer 2
        # The second convolutional layer takes as input
        # the [...] output of the first convolutional
        # layer and filters it with 256 kernels of
        # size 5 x 5 x 48
        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=96,
                out_channels=256,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        # Layer 3
        # The third convolutional layer has 384 kernels
        # of size 3x3x256 connected to the
        # outputs of the second convolutional layer
        self.conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        # Layer 4
        # The fourth convolutional layer has 384
        # kernels of size 3x3x192
        self.conv_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=384,
                out_channels=384,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )
        # Layer 5
        # the fifth convolutional layer has 256
        # kernels of size 3x3x192
        self.conv_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=384,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )

        self.linear_1 = nn.Sequential(
            nn.Linear(6144, 1024),
            nn.ReLU()
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        """
        Forward pass method. Required by nn.Module

        in: batch of data
        out:
        """
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = x.view(x.size(0), -1)
        x = self.linear_1(x)
        return self.linear_2(x)