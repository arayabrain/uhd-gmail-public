import unittest

import torch

from uhd_eeg.models.RNN.RNN import MultiLayerRNN


class TestMultiLayerRNN(unittest.TestCase):
    def setUp(
        self,
    ):
        gpu = 0
        self.device = torch.device(
            f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
        )
        self.input_size = 128
        self.hidden_size = 256
        self.num_layers = 3
        self.output_size = 5
        self.batch_size = 32
        self.duration = 320
        # LSTM
        self.model_lstm = MultiLayerRNN(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.output_size,
            rnn_type="lstm",
            bidirectional=True,
        )
        self.model_lstm.to(self.device)

        # GRU
        self.model_gru = MultiLayerRNN(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.output_size,
            rnn_type="gru",
            bidirectional=False,
        )
        self.model_gru.to(self.device)

    def test_output_shape(self):
        x = torch.randn(self.batch_size, self.duration, self.input_size)
        x = x.to(self.device)
        y_lstm = self.model_lstm(x)
        y_gru = self.model_gru(x)
        self.assertEqual(y_lstm.shape, torch.Size([self.batch_size, self.output_size]))
        self.assertEqual(y_gru.shape, torch.Size([self.batch_size, self.output_size]))


if __name__ == "__main__":
    unittest.main()
