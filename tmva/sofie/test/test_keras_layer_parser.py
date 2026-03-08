"""
Numerical validation of Exercise 5 – Keras Parser enhancements.
Tests GRU, LSTM, and Conv2DTranspose weight extraction and gate reordering.

Does NOT require ROOT. Uses only Keras, NumPy, and PyTorch (for Conv2DTranspose cross-check).

Weight-extraction helpers below replicate the logic added to parser.py so that
the tests serve as a ground-truth cross-check independent of the ROOT integration.

Run with:
    source /media/harsh/MyPassport1/root-venv/bin/activate
    cd /media/harsh/MyPassport1/root/tmva/sofie/test
    python test_keras_layer_parser.py
"""
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import keras

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
errors = []

def check(condition, msg):
    if condition:
        print(f"  {PASS}  {msg}")
    else:
        print(f"  {FAIL}  {msg}")
        errors.append(msg)

def allclose(a, b, atol=1e-5, msg=""):
    ok = np.allclose(a, b, atol=atol)
    if not ok:
        print(f"         max diff = {np.abs(a - b).max():.3e}  [{msg}]")
    return ok

# ─────────────────────────────────────────────────────────────────────────────
# Weight-extraction helpers  (mirror parser.py logic, ROOT-free)
# ─────────────────────────────────────────────────────────────────────────────

def extract_keras_lstm_weights(layer):
    """
    Convert Keras LSTM weights to ONNX LSTM format.

    Keras layout:
      kernel            [input, 4H]  gate order IFCO
      recurrent_kernel  [H,     4H]  gate order IFCO
      bias              [4H]         gate order IFCO

    ONNX layout:
      W  [1, 4H, input]  gate order IOFC
      R  [1, 4H, H]      gate order IOFC
      B  [1, 8H]         [W_biases_IOFC | R_biases_IOFC] (R biases = zeros)
    """
    ws = layer.get_weights()         # [kernel, recurrent_kernel, bias]
    kernel   = ws[0].copy()          # [input, 4H]  IFCO
    rec_kern = ws[1].copy()          # [H,     4H]  IFCO
    bias     = ws[2].copy() if len(ws) > 2 else np.zeros(kernel.shape[1])  # [4H]

    H = layer.units

    def _reorder_ifco_to_iofc(arr, axis):
        """Swap columns/rows from Keras IFCO to ONNX IOFC order."""
        # IFCO: [i, f, c, o] * H  ->  IOFC: [i, o, f, c] * H
        slices = np.split(arr, 4, axis=axis)   # i, f, c, o
        i, f, c, o = slices
        return np.concatenate([i, o, f, c], axis=axis)

    kernel   = _reorder_ifco_to_iofc(kernel,   axis=1)   # [input, 4H] IOFC
    rec_kern = _reorder_ifco_to_iofc(rec_kern, axis=1)   # [H, 4H] IOFC
    bias     = _reorder_ifco_to_iofc(bias,     axis=0)   # [4H] IOFC

    # Transpose kernels: [input, 4H] -> [4H, input], then prepend num_directions=1
    W = kernel.T[np.newaxis, :, :]       # [1, 4H, input]
    R = rec_kern.T[np.newaxis, :, :]     # [1, 4H, H]
    # ONNX B = [W_biases | R_biases]; Keras has only input bias, R biases = zeros
    B = np.concatenate([bias, np.zeros_like(bias)])[np.newaxis, :]  # [1, 8H]

    return W, R, B


def extract_keras_gru_weights(layer):
    """
    Convert Keras GRU weights to ONNX GRU format.

    Keras layout (reset_after=True, default):
      kernel            [input, 3H]  gate order ZRH  (= ONNX order, no reorder needed)
      recurrent_kernel  [H,     3H]  gate order ZRH
      bias              [2, 3H]      bias[0]=input-bias, bias[1]=recurrent-bias

    Keras layout (reset_after=False):
      bias              [3H]         only input bias

    ONNX layout:
      W  [1, 3H, input]  ZRH
      R  [1, 3H, H]      ZRH
      B  [1, 6H]         [W_biases_ZRH | R_biases_ZRH]
    """
    ws = layer.get_weights()
    kernel   = ws[0].copy()   # [input, 3H]  ZRH
    rec_kern = ws[1].copy()   # [H,     3H]  ZRH
    bias_raw = ws[2].copy() if len(ws) > 2 else None

    # No gate reordering needed: Keras GRU = ZRH = ONNX ZRH

    # Transpose kernels
    W = kernel.T[np.newaxis, :, :]       # [1, 3H, input]
    R = rec_kern.T[np.newaxis, :, :]     # [1, 3H, H]

    # Bias: [2, 3H] -> [6H] or [3H] -> [6H] with zero recurrent part
    if bias_raw is None:
        H = layer.units
        b_flat = np.zeros(6 * H, dtype=np.float32)
    elif bias_raw.ndim == 2:              # reset_after=True: [2, 3H]
        b_flat = bias_raw.flatten()       # [6H] = [b_input | b_recurrent] in ZRH
    else:                                 # reset_after=False: [3H]
        b_flat = np.concatenate([bias_raw, np.zeros_like(bias_raw)])

    B = b_flat[np.newaxis, :]            # [1, 6H]
    return W, R, B


def extract_keras_conv2dtranspose_kernel(layer):
    """
    Convert Keras Conv2DTranspose kernel to ONNX ConvTranspose weight format.

    Keras kernel shape: [kH, kW, C_out, C_in]
    ONNX weight shape:  [C_in, C_out, kH, kW]  (via transpose (3,2,0,1))
    """
    kernel = layer.get_weights()[0]      # [kH, kW, C_out, C_in]
    bias   = layer.get_weights()[1] if len(layer.get_weights()) > 1 else None
    return kernel.transpose(3, 2, 0, 1), bias   # [C_in, C_out, kH, kW]


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: LSTM – gate reordering IFCO -> IOFC + numerical inference
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Test 1: Keras LSTM weight extraction and gate reordering ---")

np.random.seed(42)
INPUT_SIZE, HIDDEN_SIZE, SEQ_LEN, BATCH = 4, 6, 3, 1

lstm_layer = keras.layers.LSTM(HIDDEN_SIZE)
k_model = keras.Sequential([keras.Input(shape=(SEQ_LEN, INPUT_SIZE)), lstm_layer])

# Verify gate reordering with known weights
ws = lstm_layer.get_weights()
# Set kernel bias to marker values: [i=1, f=2, c=3, o=4] per gate group
ws[2] = np.repeat([1., 2., 3., 4.], HIDDEN_SIZE).astype(np.float32)
lstm_layer.set_weights(ws)

W_onnx, R_onnx, B_onnx = extract_keras_lstm_weights(lstm_layer)

# ONNX IOFC order: b[0:H]=i, b[H:2H]=o, b[2H:3H]=f, b[3H:4H]=c
H = HIDDEN_SIZE
b_vals = B_onnx[0, :4*H]   # first half = W biases
check(np.allclose(b_vals[0:H],   1.), "LSTM bias i-gate = 1.0 (input gate, stays at pos 0)")
check(np.allclose(b_vals[H:2*H], 4.), "LSTM bias o-gate = 4.0 (output gate, moved to pos 1)")
check(np.allclose(b_vals[2*H:3*H], 2.), "LSTM bias f-gate = 2.0 (forget gate, moved to pos 2)")
check(np.allclose(b_vals[3*H:4*H], 3.), "LSTM bias c-gate = 3.0 (cell gate, moved to pos 3)")

check(W_onnx.shape == (1, 4*H, INPUT_SIZE), f"LSTM W shape {W_onnx.shape} == (1, {4*H}, {INPUT_SIZE})")
check(R_onnx.shape == (1, 4*H, H),           f"LSTM R shape {R_onnx.shape} == (1, {4*H}, {H})")
check(B_onnx.shape == (1, 8*H),              f"LSTM B shape {B_onnx.shape} == (1, {8*H})")
check(np.allclose(B_onnx[0, 4*H:], 0.),     "LSTM recurrent bias = zeros (appended)")

# Numerical forward-pass validation: re-randomise weights, run Keras, run manual with ONNX weights
np.random.seed(7)
ws_rand = [np.random.randn(*w.shape).astype(np.float32) for w in lstm_layer.get_weights()]
lstm_layer.set_weights(ws_rand)
keras.backend.clear_session()

x_np = np.random.randn(BATCH, SEQ_LEN, INPUT_SIZE).astype(np.float32)
keras_out = k_model(x_np).numpy()    # [batch, H]  (return_sequences=False)

W_onnx, R_onnx, B_onnx = extract_keras_lstm_weights(lstm_layer)
Wi, Wo, Wf, Wg = np.split(W_onnx[0], 4, axis=0)   # each [H, input]
Ri, Ro, Rf, Rg = np.split(R_onnx[0], 4, axis=0)   # each [H, H]
b_W = B_onnx[0, :4*H]; b_R = B_onnx[0, 4*H:]
bWi,bWo,bWf,bWg = b_W[0:H], b_W[H:2*H], b_W[2*H:3*H], b_W[3*H:4*H]
bRi,bRo,bRf,bRg = b_R[0:H], b_R[H:2*H], b_R[2*H:3*H], b_R[3*H:4*H]

h = np.zeros(H); c = np.zeros(H)
for t in range(SEQ_LEN):
    xt = x_np[0, t]
    i_g = 1/(1+np.exp(-(xt@Wi.T + bWi + h@Ri.T + bRi)))
    f_g = 1/(1+np.exp(-(xt@Wf.T + bWf + h@Rf.T + bRf)))
    g_g = np.tanh(xt@Wg.T + bWg + h@Rg.T + bRg)
    o_g = 1/(1+np.exp(-(xt@Wo.T + bWo + h@Ro.T + bRo)))
    c = f_g * c + i_g * g_g
    h = o_g * np.tanh(c)

check(allclose(h, keras_out[0], msg="LSTM"), "LSTM numerical output matches Keras (IFCO->IOFC validated)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: GRU – bias handling [2,3H]->1,6H + numerical inference
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Test 2: Keras GRU weight extraction and bias reshape ---")

GRU_H = 6
gru_layer = keras.layers.GRU(GRU_H, reset_after=True)
g_model = keras.Sequential([keras.Input(shape=(SEQ_LEN, INPUT_SIZE)), gru_layer])

ws_g = gru_layer.get_weights()
check(ws_g[2].shape == (2, GRU_H * 3),
      f"Keras GRU bias shape is [2, 3H]=[2,{3*GRU_H}] (reset_after=True)")

W_gru, R_gru, B_gru = extract_keras_gru_weights(gru_layer)
check(W_gru.shape == (1, 3*GRU_H, INPUT_SIZE), "GRU W shape (1, 3H, input)")
check(R_gru.shape == (1, 3*GRU_H, GRU_H),      "GRU R shape (1, 3H, H)")
check(B_gru.shape == (1, 6*GRU_H),             "GRU B shape (1, 6H)")

# Verify bias content: bias[0] and bias[1] concatenated
expected_B = ws_g[2].flatten()[np.newaxis, :]
check(np.allclose(B_gru, expected_B), "GRU B = [input_bias | recurrent_bias] flattened")

# Numerical forward-pass
np.random.seed(13)
ws_rand_g = [np.random.randn(*w.shape).astype(np.float32) for w in gru_layer.get_weights()]
gru_layer.set_weights(ws_rand_g)
keras.backend.clear_session()

x_np_g = np.random.randn(BATCH, SEQ_LEN, INPUT_SIZE).astype(np.float32)
keras_out_g = g_model(x_np_g).numpy()   # [batch, GRU_H]

W_gru, R_gru, B_gru = extract_keras_gru_weights(gru_layer)
Wz, Wr, Wh = np.split(W_gru[0], 3, axis=0)   # each [GRU_H, input]
Rz, Rr, Rh = np.split(R_gru[0], 3, axis=0)   # each [GRU_H, GRU_H]
H = GRU_H
bWz,bWr,bWh = B_gru[0,0:H], B_gru[0,H:2*H], B_gru[0,2*H:3*H]  # input biases (ZRH)
bRz,bRr,bRh = B_gru[0,3*H:4*H], B_gru[0,4*H:5*H], B_gru[0,5*H:]  # recurrent biases

h = np.zeros(H)
for t in range(SEQ_LEN):
    xt = x_np_g[0, t]
    z = 1/(1+np.exp(-(xt@Wz.T + bWz + h@Rz.T + bRz)))
    r = 1/(1+np.exp(-(xt@Wr.T + bWr + h@Rr.T + bRr)))
    # linear_before_reset=1 (Keras reset_after=True): r applied after linear(h)
    n = np.tanh(xt@Wh.T + bWh + r*(h@Rh.T + bRh))
    h = (1 - z)*n + z*h

check(allclose(h, keras_out_g[0], msg="GRU"), "GRU numerical output matches Keras (linear_before_reset=1)")

# GRU reset_after=False (no recurrent bias in Keras)
gru_no_reset = keras.layers.GRU(GRU_H, reset_after=False)
g_model2 = keras.Sequential([keras.Input(shape=(SEQ_LEN, INPUT_SIZE)), gru_no_reset])
ws_nr = gru_no_reset.get_weights()
check(ws_nr[2].shape == (3*GRU_H,), "Keras GRU bias shape is [3H] when reset_after=False")
W2, R2, B2 = extract_keras_gru_weights(gru_no_reset)
check(B2.shape == (1, 6*GRU_H), "GRU (no reset_after) B padded to [1, 6H] with zeros")
check(np.allclose(B2[0, 3*H:], 0.), "GRU (no reset_after) recurrent bias = zeros")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Conv2DTranspose – weight transpose (3,2,0,1) cross-checked with PyTorch
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Test 3: Conv2DTranspose weight extraction (cross-checked with PyTorch) ---")

IN_CH, OUT_CH, KH, KW = 4, 8, 3, 3

# ── Explicit-index test: verify transpose axes (3,2,0,1) ──────────────────
#   Keras kernel [kH,kW,C_out,C_in], set one known entry, confirm ONNX layout.
ct_layer = keras.layers.Conv2DTranspose(OUT_CH, (KH, KW), strides=(1,1), padding='valid', use_bias=True)
ct_model = keras.Sequential([keras.Input(shape=(8, 8, IN_CH)), ct_layer])

k_known = np.zeros((KH, KW, OUT_CH, IN_CH), dtype=np.float32)
k_known[1, 2, 3, 0] = 7.0   # kH=1, kW=2, C_out=3, C_in=0
ct_layer.set_weights([k_known, np.zeros(OUT_CH, dtype=np.float32)])
onnx_k, onnx_b = extract_keras_conv2dtranspose_kernel(ct_layer)

check(onnx_k.shape == (IN_CH, OUT_CH, KH, KW),
      f"Conv2DTranspose kernel shape {onnx_k.shape} == ({IN_CH},{OUT_CH},{KH},{KW})")
check(onnx_b.shape == (OUT_CH,), f"Conv2DTranspose bias shape {onnx_b.shape}")
# After (3,2,0,1): [C_in=0, C_out=3, kH=1, kW=2] should be 7.0
check(onnx_k[0, 3, 1, 2] == 7.0,
      "kernel transpose (3,2,0,1): [kH,kW,Cout,Cin]->[Cin,Cout,kH,kW] correct")
# All other entries should be 0
check(np.sum(np.abs(onnx_k)) == 7.0, "only the one known entry survives transpose")

# ── Numerical validation: padding='valid', stride=1 ───────────────────────
#   With valid padding and stride=1, output shape is unambiguous:
#   H_out = H_in + kH - 1 = 8 + 3 - 1 = 10.
#   PyTorch ConvTranspose2d with padding=0 implements exactly this.
np.random.seed(99)
k_v = np.random.randn(KH, KW, OUT_CH, IN_CH).astype(np.float32)
k_b = np.random.randn(OUT_CH).astype(np.float32)
ct_layer.set_weights([k_v, k_b])

x_nhwc = np.random.randn(1, 8, 8, IN_CH).astype(np.float32)
keras_ct_out = ct_model(x_nhwc).numpy()   # [1, 10, 10, OUT_CH] NHWC

onnx_kv, onnx_bv = extract_keras_conv2dtranspose_kernel(ct_layer)
pt_conv = nn.ConvTranspose2d(IN_CH, OUT_CH, (KH, KW), stride=1, padding=0, bias=True)
with torch.no_grad():
    pt_conv.weight.data = torch.from_numpy(onnx_kv)  # [C_in, C_out, kH, kW]
    pt_conv.bias.data   = torch.from_numpy(onnx_bv)

x_nchw = torch.from_numpy(x_nhwc.transpose(0, 3, 1, 2))   # NHWC -> NCHW
pt_out  = pt_conv(x_nchw).detach().numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC

check(pt_out.shape == keras_ct_out.shape, f"Conv2DTranspose output shape {pt_out.shape}")
check(allclose(pt_out, keras_ct_out, atol=1e-4, msg="Conv2DTranspose valid"),
      "Conv2DTranspose padding='valid': PyTorch(ONNX weights) == Keras output")

# ── Shape check for padding='same' ────────────────────────────────────────
#   We only verify shapes here because 'same' padding symmetry rules differ
#   between frameworks.  The weight extraction is identical – only autopad tag changes.
ct_same = keras.layers.Conv2DTranspose(OUT_CH, (KH, KW), strides=(2,2), padding='same', use_bias=False)
ct_s_model = keras.Sequential([keras.Input(shape=(8, 8, IN_CH)), ct_same])
ct_same.set_weights([np.ones((KH, KW, OUT_CH, IN_CH), dtype=np.float32)])
onnx_ks, _ = extract_keras_conv2dtranspose_kernel(ct_same)
check(onnx_ks.shape == (IN_CH, OUT_CH, KH, KW),
      "Conv2DTranspose padding='same' kernel shape correct")


# ─────────────────────────────────────────────────────────────────────────────
# Final result
# ─────────────────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"RESULT: {len(errors)} test(s) FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("RESULT: All tests PASSED ✓")
    sys.exit(0)
