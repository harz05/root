"""
Exercise 4 Demo: PyTorch Layer Parser for SOFIE
================================================
Demonstrates parsing of ELU, MaxPool2D, BatchNorm2D, RNN, LSTM, GRU
from real PyTorch models, showing extracted weights and node info.

Run: python3 demo_pytorch_layer_parser.py
"""

import torch
import torch.nn as nn
import numpy as np
from PyTorchLayerParser import (
    parse_elu, parse_maxpool2d, parse_batchnorm2d,
    parse_rnn, parse_lstm, parse_gru,
    parse_model, print_parsed_model
)

torch.manual_seed(42)

# ----------------------------------------------------------------
# Demo 1: Individual parsers
# ----------------------------------------------------------------
print("=" * 60)
print("Demo 1: Individual Layer Parsers")
print("=" * 60)

# ELU
elu_module = nn.ELU(alpha=2.0)
elu_node   = parse_elu(elu_module, 'input', 'elu_out')
print(f"\nELU  -> nodeType: {elu_node['nodeType']}, alpha={elu_node['nodeAttributes']['alpha']}")

# MaxPool2D
mp_module = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)
mp_node   = parse_maxpool2d(mp_module, 'conv_out', 'pool_out')
print(f"MaxPool2D -> nodeType: {mp_node['nodeType']}")
print(f"  kernel_shape={mp_node['nodeAttributes']['kernel_shape']}, "
      f"strides={mp_node['nodeAttributes']['strides']}, "
      f"pads={mp_node['nodeAttributes']['pads']}")

# BatchNorm2D
bn_module = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
bn_module.eval()
bn_node   = parse_batchnorm2d(bn_module, 'relu_out', 'bn_out', 'bn_layer')
print(f"BatchNorm2D -> nodeType: {bn_node['nodeType']}")
for wname, wdata in bn_node['weights'].items():
    print(f"  weight '{wname}': shape={wdata.shape}")

# RNN
rnn_module = nn.RNN(input_size=16, hidden_size=32)
rnn_node   = parse_rnn(rnn_module, 'seq_in', 'rnn_out', 'rnn_layer')
print(f"\nRNN  -> nodeType: {rnn_node['nodeType']}, "
      f"hidden_size={rnn_node['nodeAttributes']['hidden_size']}")
for wname, wdata in rnn_node['weights'].items():
    print(f"  weight '{wname}': shape={wdata.shape}")

# LSTM
lstm_module = nn.LSTM(input_size=16, hidden_size=32)
lstm_node   = parse_lstm(lstm_module, 'seq_in', 'lstm_out', 'lstm_layer')
print(f"\nLSTM -> nodeType: {lstm_node['nodeType']}, "
      f"hidden_size={lstm_node['nodeAttributes']['hidden_size']}")
for wname, wdata in lstm_node['weights'].items():
    print(f"  weight '{wname}': shape={wdata.shape}")
print(f"  outputs: {lstm_node['nodeOutputs']}")

# GRU
gru_module = nn.GRU(input_size=16, hidden_size=32)
gru_node   = parse_gru(gru_module, 'seq_in', 'gru_out', 'gru_layer')
print(f"\nGRU  -> nodeType: {gru_node['nodeType']}, "
      f"hidden_size={gru_node['nodeAttributes']['hidden_size']}")
for wname, wdata in gru_node['weights'].items():
    print(f"  weight '{wname}': shape={wdata.shape}")
print(f"  outputs: {gru_node['nodeOutputs']}")


# ----------------------------------------------------------------
# Demo 2: CNN model with BatchNorm + ELU + MaxPool
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("Demo 2: CNN Model with BatchNorm + ELU + MaxPool")
print("=" * 60)

class CNNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv   = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn     = nn.BatchNorm2d(16)
        self.elu    = nn.ELU(alpha=1.0)
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.elu(self.bn(self.conv(x))))

cnn = CNNBlock()
cnn.eval()
parsed_cnn = parse_model(cnn, (3, 64, 64))
print_parsed_model(parsed_cnn)


# ----------------------------------------------------------------
# Demo 3: Bidirectional LSTM model
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("Demo 3: Bidirectional LSTM")
print("=" * 60)

class BiLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=16, bidirectional=True)

    def forward(self, x):
        y, _ = self.lstm(x)
        return y

bilstm = BiLSTMModel()
bilstm.eval()
parsed_bilstm = parse_model(bilstm, (10, 1, 8))
print_parsed_model(parsed_bilstm)

# Verify both directions are captured
W = parsed_bilstm['weights']['lstm_W']
print(f"Bidirectional LSTM W shape: {W.shape}  (expected [2, 64, 8])")


# ----------------------------------------------------------------
# Demo 4: Stacked RNN -> GRU model
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("Demo 4: Stacked RNN + GRU")
print("=" * 60)

class StackedRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=4, hidden_size=8)
        self.gru = nn.GRU(input_size=8, hidden_size=4)

    def forward(self, x):
        y, _ = self.rnn(x)
        y, _ = self.gru(y)
        return y

stacked = StackedRNNModel()
stacked.eval()
parsed_stacked = parse_model(stacked, (5, 1, 4))
print_parsed_model(parsed_stacked)


# ----------------------------------------------------------------
# Demo 5: LSTM gate reordering validation (IFGO -> IOFC)
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("Demo 5: LSTM Gate Reordering Validation")
print("=" * 60)

torch.manual_seed(7)
lstm_check = nn.LSTM(input_size=4, hidden_size=4)
node_check  = parse_lstm(lstm_check, 'x', 'y', 'lstm_check')

# Run PyTorch LSTM
x_pt    = torch.randn(3, 1, 4)
out_pt, (h_pt, c_pt) = lstm_check(x_pt)

# Manually verify gate extraction is consistent with PyTorch forward pass
W_onnx = node_check['weights']['lstm_check_W'][0]  # [4H, input]
H = 4
# ONNX [i,o,f,g] -> split
i_o, o_o, f_o, g_o = np.split(W_onnx, 4, axis=0)
# PyTorch [i,f,g,o] -> split
w_ih = lstm_check.weight_ih_l0.detach().numpy()
i_p, f_p, g_p, o_p = np.split(w_ih, 4, axis=0)

match_i = np.allclose(i_o, i_p, atol=1e-6)
match_o = np.allclose(o_o, o_p, atol=1e-6)
match_f = np.allclose(f_o, f_p, atol=1e-6)
match_g = np.allclose(g_o, g_p, atol=1e-6)
print(f"  Gate i (input)  correctly mapped: {match_i}")
print(f"  Gate o (output) correctly mapped: {match_o}")
print(f"  Gate f (forget) correctly mapped: {match_f}")
print(f"  Gate g (cell)   correctly mapped: {match_g}")

all_ok = all([match_i, match_o, match_f, match_g])
print(f"\n  LSTM gate reordering: {'CORRECT ✓' if all_ok else 'INCORRECT ✗'}")

print("\n" + "=" * 60)
print("All demos complete.")
print("=" * 60)
