"""
Test suite for PyTorchLayerParser.py  (GSoC 2026 SOFIE Exercise 4)

Tests each of the 6 layer parsers:
  1. ELU
  2. MaxPool2D
  3. BatchNorm2D
  4. RNN
  5. LSTM
  6. GRU

For every layer we verify:
  - Correct nodeType (ONNX name)
  - Correct nodeAttributes
  - Correct weight tensor names, shapes, and values
  - Correctness of weight reordering (LSTM IFGO->IOFC, GRU RZN->ZRH)
  - parse_model() integration on composite models
"""

import sys
import numpy as np
import torch
import torch.nn as nn

# Adjust path if running from a different directory
sys.path.insert(0, '.')
from PyTorchLayerParser import (
    parse_elu, parse_maxpool2d, parse_batchnorm2d,
    parse_rnn, parse_lstm, parse_gru,
    parse_model, print_parsed_model
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

errors = []

def check(condition, msg):
    if condition:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  {msg}")
        errors.append(msg)

# ============================================================
# 1. ELU
# ============================================================
print("\n--- Test 1: ELU ---")

elu = nn.ELU(alpha=0.5)
node = parse_elu(elu, 'x', 'y')

check(node['nodeType'] == 'onnx::Elu',       "nodeType is onnx::Elu")
check(abs(node['nodeAttributes']['alpha'] - 0.5) < 1e-6, "alpha = 0.5")
check(node['nodeInputs']  == ['x'],           "input tensor name")
check(node['nodeOutputs'] == ['y'],           "output tensor name")
check(len(node['weights']) == 0,              "no weight tensors (correct)")

# Default alpha
elu_default = nn.ELU()
node_default = parse_elu(elu_default, 'x', 'y')
check(abs(node_default['nodeAttributes']['alpha'] - 1.0) < 1e-6, "default alpha = 1.0")

# ============================================================
# 2. MaxPool2D
# ============================================================
print("\n--- Test 2: MaxPool2D ---")

mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
node = parse_maxpool2d(mp, 'x', 'y')

check(node['nodeType'] == 'onnx::MaxPool',      "nodeType is onnx::MaxPool")
check(node['nodeAttributes']['kernel_shape'] == [3, 3],  "kernel_shape = [3,3]")
check(node['nodeAttributes']['strides']      == [2, 2],  "strides = [2,2]")
check(node['nodeAttributes']['pads']         == [1, 1, 1, 1], "pads = [1,1,1,1]")
check(len(node['weights']) == 0,                "no weight tensors (correct)")

# Asymmetric kernel
mp2 = nn.MaxPool2d(kernel_size=(2, 3), stride=(1, 2))
node2 = parse_maxpool2d(mp2, 'x', 'y')
check(node2['nodeAttributes']['kernel_shape'] == [2, 3], "asymmetric kernel [2,3]")
check(node2['nodeAttributes']['strides']      == [1, 2], "asymmetric stride [1,2]")

# ============================================================
# 3. BatchNorm2D
# ============================================================
print("\n--- Test 3: BatchNorm2D ---")

torch.manual_seed(42)
bn = nn.BatchNorm2d(16, eps=1e-4, momentum=0.05)
bn.eval()

node = parse_batchnorm2d(bn, 'x', 'y', 'bn0')

check(node['nodeType'] == 'onnx::BatchNormalization', "nodeType is onnx::BatchNormalization")
check(abs(node['nodeAttributes']['epsilon']  - 1e-4) < 1e-9, "epsilon = 1e-4")
check(abs(node['nodeAttributes']['momentum'] - 0.05) < 1e-9, "momentum = 0.05")
check(node['nodeAttributes']['training_mode'] == 0,           "training_mode = 0 (inference)")

w = node['weights']
check('bn0_scale'        in w, "scale weight present")
check('bn0_bias'         in w, "bias weight present")
check('bn0_running_mean' in w, "running_mean present")
check('bn0_running_var'  in w, "running_var present")

check(w['bn0_scale'].shape        == (16,), "scale shape (16,)")
check(w['bn0_bias'].shape         == (16,), "bias shape (16,)")
check(w['bn0_running_mean'].shape == (16,), "running_mean shape (16,)")
check(w['bn0_running_var'].shape  == (16,), "running_var shape (16,)")

# Verify values match module
np.testing.assert_allclose(w['bn0_scale'],        bn.weight.detach().numpy(),      rtol=1e-5)
np.testing.assert_allclose(w['bn0_bias'],         bn.bias.detach().numpy(),        rtol=1e-5)
np.testing.assert_allclose(w['bn0_running_mean'], bn.running_mean.detach().numpy(),rtol=1e-5)
np.testing.assert_allclose(w['bn0_running_var'],  bn.running_var.detach().numpy(), rtol=1e-5)
check(True, "BatchNorm2D weight values match module parameters")

# ============================================================
# 4. RNN
# ============================================================
print("\n--- Test 4: RNN ---")

torch.manual_seed(0)
rnn = nn.RNN(input_size=8, hidden_size=16, nonlinearity='tanh', bias=True)

node = parse_rnn(rnn, 'x', 'y', 'rnn0')

check(node['nodeType'] == 'onnx::RNN', "nodeType is onnx::RNN")
check(node['nodeAttributes']['hidden_size'] == 16,       "hidden_size = 16")
check(node['nodeAttributes']['activations'] == ['TANH'], "activation = TANH")
check(node['nodeAttributes']['direction']   == 'forward',"direction = forward")

w = node['weights']
check('rnn0_W' in w, "W weight present")
check('rnn0_R' in w, "R weight present")
check('rnn0_B' in w, "B weight present")

check(w['rnn0_W'].shape == (1, 16,  8), "W shape (1, hidden, input)")
check(w['rnn0_R'].shape == (1, 16, 16), "R shape (1, hidden, hidden)")
check(w['rnn0_B'].shape == (1, 32),     "B shape (1, 2*hidden)")

# Verify W values
np.testing.assert_allclose(w['rnn0_W'][0], rnn.weight_ih_l0.detach().numpy(), rtol=1e-5)
np.testing.assert_allclose(w['rnn0_R'][0], rnn.weight_hh_l0.detach().numpy(), rtol=1e-5)
check(True, "RNN weight values match module parameters")

# RELU activation variant
rnn_relu = nn.RNN(input_size=4, hidden_size=8, nonlinearity='relu')
node_relu = parse_rnn(rnn_relu, 'x', 'y', 'rnn_relu')
check(node_relu['nodeAttributes']['activations'] == ['RELU'], "relu activation captured")

# Bidirectional
rnn_bi = nn.RNN(input_size=4, hidden_size=8, bidirectional=True)
node_bi = parse_rnn(rnn_bi, 'x', 'y', 'rnn_bi')
check(node_bi['nodeAttributes']['direction']    == 'bidirectional',  "bidirectional direction")
check(node_bi['weights']['rnn_bi_W'].shape == (2, 8, 4),             "bidir W shape (2, H, I)")

# ============================================================
# 5. LSTM
# ============================================================
print("\n--- Test 5: LSTM ---")

torch.manual_seed(1)
lstm = nn.LSTM(input_size=10, hidden_size=20, bias=True)

node = parse_lstm(lstm, 'x', 'y', 'lstm0')

check(node['nodeType'] == 'onnx::LSTM', "nodeType is onnx::LSTM")
check(node['nodeAttributes']['hidden_size'] == 20, "hidden_size = 20")
check(node['nodeAttributes']['direction']   == 'forward', "direction = forward")

w = node['weights']
check('lstm0_W' in w, "W weight present")
check('lstm0_R' in w, "R weight present")
check('lstm0_B' in w, "B weight present")

check(w['lstm0_W'].shape == (1, 80, 10), "W shape (1, 4*H, input)")
check(w['lstm0_R'].shape == (1, 80, 20), "R shape (1, 4*H, H)")
check(w['lstm0_B'].shape == (1, 160),    "B shape (1, 8*H)")

# Verify IFGO -> IOFC gate reordering
H = 20
w_ih = lstm.weight_ih_l0.detach().numpy()   # PyTorch: [i, f, g, o] x H rows
i_pt, f_pt, g_pt, o_pt = np.split(w_ih, 4, axis=0)
# Expected ONNX order [i, o, f, g]:
expected_W0 = np.concatenate([i_pt, o_pt, f_pt, g_pt], axis=0)
np.testing.assert_allclose(w['lstm0_W'][0], expected_W0, rtol=1e-5)
check(True, "LSTM gate reordering IFGO->IOFC correct")

# Outputs should include Y, Y_h, Y_c
check('y_Y'   in node['nodeOutputs'], "Y output present")
check('y_Y_h' in node['nodeOutputs'], "Y_h output present")
check('y_Y_c' in node['nodeOutputs'], "Y_c output present")

# Bidirectional LSTM
lstm_bi = nn.LSTM(input_size=4, hidden_size=8, bidirectional=True)
node_bi = parse_lstm(lstm_bi, 'x', 'y', 'lstm_bi')
check(node_bi['weights']['lstm_bi_W'].shape == (2, 32, 4), "bidir LSTM W shape (2, 4H, I)")

# ============================================================
# 6. GRU
# ============================================================
print("\n--- Test 6: GRU ---")

torch.manual_seed(2)
gru = nn.GRU(input_size=6, hidden_size=12, bias=True)

node = parse_gru(gru, 'x', 'y', 'gru0')

check(node['nodeType'] == 'onnx::GRU', "nodeType is onnx::GRU")
check(node['nodeAttributes']['hidden_size'] == 12, "hidden_size = 12")
check(node['nodeAttributes']['direction']   == 'forward', "direction = forward")

w = node['weights']
check('gru0_W' in w, "W weight present")
check('gru0_R' in w, "R weight present")
check('gru0_B' in w, "B weight present")

check(w['gru0_W'].shape == (1, 36,  6), "W shape (1, 3*H, input)")
check(w['gru0_R'].shape == (1, 36, 12), "R shape (1, 3*H, H)")
check(w['gru0_B'].shape == (1, 72),     "B shape (1, 6*H)")

# Verify RZN -> ZRH gate reordering
H = 12
w_ih = gru.weight_ih_l0.detach().numpy()   # PyTorch: [r, z, n] x H rows
r_pt, z_pt, n_pt = np.split(w_ih, 3, axis=0)
# Expected ONNX order [z, r, n]:
expected_W0 = np.concatenate([z_pt, r_pt, n_pt], axis=0)
np.testing.assert_allclose(w['gru0_W'][0], expected_W0, rtol=1e-5)
check(True, "GRU gate reordering RZN->ZRH correct")

# Outputs should include Y, Y_h
check('y_Y'   in node['nodeOutputs'], "Y output present")
check('y_Y_h' in node['nodeOutputs'], "Y_h output present")

# Bidirectional GRU
gru_bi = nn.GRU(input_size=4, hidden_size=8, bidirectional=True)
node_bi = parse_gru(gru_bi, 'x', 'y', 'gru_bi')
check(node_bi['weights']['gru_bi_W'].shape == (2, 24, 4), "bidir GRU W shape (2, 3H, I)")

# ============================================================
# 7. parse_model() integration test
# ============================================================
print("\n--- Test 7: parse_model() integration ---")

class MixedModel(nn.Module):
    """A model combining ELU, MaxPool2d, BatchNorm2d, and GRU."""
    def __init__(self):
        super().__init__()
        self.conv  = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn    = nn.BatchNorm2d(8)
        self.elu   = nn.ELU(alpha=0.3)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.pool(x)
        return x

class RecurrentModel(nn.Module):
    """Model with LSTM and GRU stacked."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=16, hidden_size=32)
        self.gru  = nn.GRU(input_size=32, hidden_size=16)

    def forward(self, x):
        y, _ = self.lstm(x)
        y, _ = self.gru(y)
        return y

torch.manual_seed(99)
mixed = MixedModel()
parsed = parse_model(mixed, (3, 32, 32))

bn_nodes  = [n for n in parsed['nodes'] if n['nodeType'] == 'onnx::BatchNormalization']
elu_nodes  = [n for n in parsed['nodes'] if n['nodeType'] == 'onnx::Elu']
pool_nodes = [n for n in parsed['nodes'] if n['nodeType'] == 'onnx::MaxPool']

check(len(bn_nodes)  == 1, "MixedModel: BatchNorm2D parsed")
check(len(elu_nodes) == 1, "MixedModel: ELU parsed")
check(len(pool_nodes)== 1, "MixedModel: MaxPool2D parsed")
check(len(parsed['weights']) > 0, "MixedModel: weights extracted")

recurrent = RecurrentModel()
parsed_rec = parse_model(recurrent, (10, 1, 16))
lstm_nodes = [n for n in parsed_rec['nodes'] if n['nodeType'] == 'onnx::LSTM']
gru_nodes  = [n for n in parsed_rec['nodes'] if n['nodeType'] == 'onnx::GRU']
check(len(lstm_nodes) == 1, "RecurrentModel: LSTM parsed")
check(len(gru_nodes)  == 1, "RecurrentModel: GRU parsed")

print("\n--- Parsed Model Summary (MixedModel) ---")
print_parsed_model(parsed)

# ============================================================
# Final result
# ============================================================
print()
if errors:
    print(f"RESULT: {len(errors)} test(s) FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("RESULT: All tests PASSED ✓")
    sys.exit(0)
