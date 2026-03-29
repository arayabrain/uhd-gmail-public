"""
This is RNN implementation.
"""

from typing import Optional

import torch
import torch.nn as nn


class MultiLayerRNN(nn.Module):
    """MultiLayerRNN"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        dropout_rate: float = 0.5,
        last_activation: Optional[str] = None,
    ) -> None:
        """__init__

        Args:
            input_size (int): input size
            hidden_size (int): hidden size
            num_layers (int): number of layers
            output_size (int): output size
            rnn_type (str, optional): rnn type. Defaults to "lstm".
            bidirectional (bool, optional): bidirectional or not. Defaults to False.
            dropout_rate (float, optional): drop out rate. Defaults to 0.5.
            last_activation (Optional[str], optional): last layer activation func. Defaults to None.
        """
        super(MultiLayerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate

        # define RNN layers
        if self.rnn_type == "lstm":
            rnn_cell = nn.LSTM
        elif self.rnn_type == "gru":
            rnn_cell = nn.GRU
        else:
            raise ValueError("Invalid RNN type. Use 'lstm' or 'gru'.")

        self.D = 2 if self.bidirectional else 1
        self.rnn = rnn_cell(
            input_size,
            hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
            dropout=self.dropout_rate,
        )
        # define output layer
        self.fc = nn.Linear(hidden_size * self.D, output_size)
        if last_activation is not None:
            if last_activation == "sigmoid":
                self.last_activation = nn.Sigmoid()
            elif last_activation == "softmax":
                self.last_activation = nn.Softmax(dim=1)
            else:
                raise ValueError("Invalid last activation. Use 'sigmoid' or 'softmax'.")
        else:
            self.last_activation = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        # Generate initial hidden state (double the number of layers for bidirectional)
        h0 = torch.zeros(self.num_layers * self.D, x.size(0), self.hidden_size).to(
            x.device
        )
        # For LSTM, c0 is also initialized
        if self.rnn_type == "lstm":
            c0 = torch.zeros(self.num_layers * self.D, x.size(0), self.hidden_size).to(
                x.device
            )
        # Forward propagation of multi-layer RNNs
        if self.rnn_type == "lstm":
            x, _ = self.rnn(x, (h0, c0))
        else:
            x, _ = self.rnn(x, h0)

        # Get output at last time step
        out = self.fc(x[:, -1, :])
        if self.last_activation is not None:
            out = self.last_activation(out)
        return out
