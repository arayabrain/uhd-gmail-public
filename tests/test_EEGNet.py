import unittest

import torch
from hydra import compose, initialize

from uhd_eeg.models.CNN.EEGNet import EEGNet, EEGNet_with_mask


class TestEEGNet_with_mask(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with initialize(version_base=None, config_path="../configs/test"):
            args = compose(config_name="EEGNet_with_mask")
        self.model = EEGNet_with_mask(args, 320)
        self.model.to(self.device)
        self.n_class = args.n_class
        self.num_channels = args.num_channels
        self.channels_use = args.channels_use
        self.channels_not_use = [
            i for i in range(self.num_channels) if i not in self.channels_use
        ]

    def test_output_shape(self):
        x = torch.randn(1, 1, self.num_channels, 320).to(self.device)
        y = self.model(x)
        masked_x = x * self.model.mask.to(self.device)
        self.assertEqual(y.shape, torch.Size([1, self.n_class]))
        self.assertNotEqual(masked_x[:, :, self.channels_use, :].sum(), torch.tensor(0))
        self.assertEqual(
            masked_x[:, :, self.channels_not_use, :].sum(), torch.tensor(0)
        )


class TestEEGNet(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with initialize(version_base=None, config_path="../configs/test"):
            args = compose(config_name="EEGNet")
        self.model = EEGNet(args, T=320)
        self.model.to(self.device)
        self.n_class = args.n_class
        self.num_channels = args.num_channels

    def test_output_shape(self):
        x = torch.randn(1, 1, self.num_channels, 320).to(self.device)
        y = self.model(x)
        self.assertEqual(y.shape, torch.Size([1, self.n_class]))


if __name__ == "__main__":
    unittest.main()
