"""CNN models"""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


class Abs(nn.Module):
    """Abs"""

    def forward(self, x: Tensor) -> Tensor:
        """forward

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        return torch.abs(x)


class HilbertTransform(nn.Module):
    """HilbertTransform"""

    def forward(self, x: Tensor) -> Tensor:
        """forward

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        Xf = torch.fft.fft(x, dim=-1)
        device = Xf.device
        N = x.size(-1)
        h = torch.zeros(N).to(device)
        h[0], h[1 : (N + 1) // 2] = 1, 2
        if N % 2 == 0:
            h[N // 2] = 1
        h = h.view(*[1] * (x.ndim - 1), -1)

        return torch.fft.ifft(Xf * h, dim=-1)


class EEGNet(nn.Module):
    """EEGNet"""

    def __init__(self, args: DictConfig, T: int) -> None:
        """__init__

        Args:
            args (DictConfig): config
            T (int): time length
        """
        super(EEGNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, args.F1, (1, args.k1), padding="same", bias=False),
            nn.BatchNorm2d(args.F1),
        )
        # TODO uncomment hilbert_transform
        self.hilbert_transform = args.hilbert_transform
        if self.hilbert_transform:
            self.hilbert_layer = nn.Sequential(
                HilbertTransform(), Abs()
            )  # Compute the spectral power using the Hilbert transform; note that this is just one of the features used in HTNet, and not used in this paper

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                args.F1,
                args.D * args.F1,
                (args.num_channels, 1),
                groups=args.F1,
                bias=False,
            ),
            nn.BatchNorm2d(args.D * args.F1),
            nn.ELU(),
            nn.AvgPool2d((1, args.p1)),
            nn.Dropout(args.dr1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                args.D * args.F1,
                args.D * args.F1,
                (1, args.k2),
                padding="same",
                groups=args.D * args.F1,
                bias=False,
            ),
            nn.Conv2d(args.D * args.F1, args.F2, (1, 1), bias=False),
            nn.BatchNorm2d(args.F2),
            nn.ELU(),
            nn.AvgPool2d((1, args.p2)),
            nn.Dropout(args.dr2),
        )

        self.n_dim = self.compute_dim(args.num_channels, T)
        self.classifier = nn.Linear(self.n_dim, args.n_class, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """forward

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        x = self.conv1(x)
        # TODO uncomment hilbert_transform
        if self.hilbert_transform:
            x = self.hilbert_layer(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.n_dim)
        x = self.classifier(x)
        return x

    def compute_dim(self, num_channels: int, T: int) -> int:
        """compute dimension

        Args:
            num_channels (int): number of channels
            T (int): time length

        Returns:
            int: dimension
        """
        x = torch.zeros((1, 1, num_channels, T))

        x = self.conv1(x)
        # TODO uncomment hilbert_transform
        if self.hilbert_transform:
            x = self.hilbert_layer(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x.size()[1] * x.size()[2] * x.size()[3]


class EEGNet_with_mask(nn.Module):
    """EEGNet_with_mask"""

    def __init__(self, args: DictConfig, T: int) -> None:
        """__init__

        Args:
            args (DictConfig): config
            T (int): time length
        """
        super(EEGNet_with_mask, self).__init__()

        # input mask for channel decimation
        self.mask = torch.zeros((1, 1, args.num_channels, T))
        self.mask[:, :, args.channels_use, :] = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, args.F1, (1, args.k1), padding="same", bias=False),
            nn.BatchNorm2d(args.F1),
        )
        self.hilbert_transform = args.hilbert_transform
        if self.hilbert_transform:
            self.hilbert_layer = nn.Sequential(
                HilbertTransform(), Abs()
            )  # Compute the spectral power using the Hilbert transform; note that this is just one of the features used in HTNet, and not used in this paper

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                args.F1,
                args.D * args.F1,
                (args.num_channels, 1),
                groups=args.F1,
                bias=False,
            ),
            nn.BatchNorm2d(args.D * args.F1),
            nn.ELU(),
            nn.AvgPool2d((1, args.p1)),
            nn.Dropout(args.dr1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                args.D * args.F1,
                args.D * args.F1,
                (1, args.k2),
                padding="same",
                groups=args.D * args.F1,
                bias=False,
            ),
            nn.Conv2d(args.D * args.F1, args.F2, (1, 1), bias=False),
            nn.BatchNorm2d(args.F2),
            nn.ELU(),
            nn.AvgPool2d((1, args.p2)),
            nn.Dropout(args.dr2),
        )

        self.n_dim = self.compute_dim(args.num_channels, T)
        self.classifier = nn.Linear(self.n_dim, args.n_class, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """forward

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        x = x * self.mask.to(x.device)
        x = self.conv1(x)
        if self.hilbert_transform:
            x = self.hilbert_layer(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.n_dim)
        x = self.classifier(x)
        return x

    def compute_dim(self, num_channels: int, T: int) -> int:
        """compute dimension

        Args:
            num_channels (int): number of channels
            T (int): time length

        Returns:
            int: dimension
        """
        x = torch.zeros((1, 1, num_channels, T))

        x = self.conv1(x)
        if self.hilbert_transform:
            x = self.hilbert_layer(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x.size()[1] * x.size()[2] * x.size()[3]
