# GSoC 2026 – SOFIE: Improving Keras and PyTorch Parsers
## Exercises 1–5 – Harsh Chauhan

---

## Exercise 1 – Build ROOT from Source with SOFIE Support

ROOT 6.39.01 was built from source with the following CMake flags to enable SOFIE
and its Python bindings:

```bash
cmake -DCMAKE_INSTALL_PREFIX=/media/harsh/MyPassport1/root-install \
      -Dtmva-sofie=ON   \
      -Dtmva-pymva=ON   \
      -Dbuiltin_protobuf=ON \
      ..
make -j$(nproc) install
```

Build verification – the key shared libraries are present:

```
libROOTTMVASofie.so          SOFIE core (code generation)
libROOTTMVASofieParser.so    ONNX + Keras + PyTorch parsers
```

The Python environment is set up via a virtualenv so that ROOT's PyROOT and Keras/PyTorch
can coexist without system-level conflicts:

```bash
python3 -m venv /media/harsh/MyPassport1/root-venv
source  /media/harsh/MyPassport1/root-venv/bin/activate
pip install tensorflow keras torch numpy
```

---

## Exercise 2 – Run Existing SOFIE Tutorials

All six TMVA/SOFIE tutorials were run successfully after the build.

| Tutorial | Status | Notes |
|---|---|---|
| `TMVA_Higgs_Classification.C` | Ran | DNN ROC 0.765, PyKeras ROC 0.760 |
| `TMVA_CNN_Classification.C`   | Ran | PyTorch CNN ROC 0.870 |
| `TMVA_SOFIE_ONNX.py`          | Ran | Linear model exported + SOFIE inference |
| `TMVA_SOFIE_PyTorch.C`        | Ran | Sequential model, C++ header generated |
| `TMVA_SOFIE_Keras.py`         | Ran | Dense+ReLU+Softmax, C++ header generated |
| `TMVA_SOFIE_Keras_HiggsModel.py` | Ran | 4-layer Higgs model, inference output correct |

Each tutorial was run end-to-end: model trained (or loaded), parsed by SOFIE,
C++ inference header generated, and the output verified.

---

## Exercise 3 – Study the Parsers and Identify Gaps

### C++ PyTorch parser coverage

The existing `RModelParser_PyTorch.cxx` handles:

- `onnx::Gemm`      → Linear layers
- `onnx::Conv`      → Conv2d layers
- `onnx::Relu`      → ReLU activation
- `onnx::Selu`      → SELU activation
- `onnx::Sigmoid`   → Sigmoid activation
- `onnx::Transpose` → Transpose operations

**Not yet supported** (the motivation for Exercises 4 and 5):

- ELU activation (`nn.ELU`)
- MaxPool2D (`nn.MaxPool2d`)
- BatchNorm2D (`nn.BatchNorm2d`)
- RNN (`nn.RNN`)
- LSTM (`nn.LSTM`) – complex because PyTorch IFGO gate order differs from ONNX IOFC
- GRU (`nn.GRU`)   – complex because PyTorch RZN gate order differs from ONNX ZRH

### Key architectural observation

The C++ parser converts a `.pt` model to an ONNX graph internally (via `_model_to_graph`),
then maps ONNX node types to `ROperator` objects. The Python-level parser in Exercise 4
mirrors this design: each `parse_*` function returns a node-info dict in the same format
as the C++ parser's `fNode` structure, making future integration into `RModelParser_PyTorch.cxx`
straightforward.

### Keras parser coverage

The existing Python Keras parser (`parser.py`) already handles Dense, Conv2D, pooling,
BatchNorm, SimpleRNN, and common activations. The following were missing or incomplete:

- `LSTM` and `GRU` – entries commented out in `mapKerasLayer`
- `Conv2DTranspose` – not implemented
- GRU bias bug – stored as `[2, 3H]` by Keras but passed to ROOT with wrong shape

---

## Exercise 4 – Implement Parsing Functions for New PyTorch Operators

### Files

| File | Description |
|---|---|
| `tmva/sofie_parsers/src/PyTorchLayerParser.py`        | Parser module – 6 operators |
| `tmva/sofie/test/test_pytorch_layer_parser.py`        | Test suite |
| `tmva/sofie/test/demo_pytorch_layer_parser.py`        | End-to-end usage demo |

### Running

```bash
source /media/harsh/MyPassport1/root-venv/bin/activate
cd /media/harsh/MyPassport1/root/tmva/sofie_parsers/src

python3 ../../sofie/test/test_pytorch_layer_parser.py
python3 ../../sofie/test/demo_pytorch_layer_parser.py
```

### Implemented operators

#### 1. ELU  (`parse_elu`)

Maps to `onnx::Elu`. Extracts the scalar `alpha` attribute. No learnable weights.

#### 2. MaxPool2D  (`parse_maxpool2d`)

Maps to `onnx::MaxPool`. Extracts `kernel_shape`, `strides`, `dilations`, `ceil_mode`,
and padding in ONNX 4-element format `[x1_begin, x2_begin, x1_end, x2_end]`.
Handles asymmetric kernels and strides.

#### 3. BatchNorm2D  (`parse_batchnorm2d`)

Maps to `onnx::BatchNormalization`. Extracts `scale` (γ), `bias` (β), `running_mean`,
`running_var`. Sets `training_mode=0` (inference). ONNX input order: `[X, scale, bias, mean, var]`.

#### 4. RNN  (`parse_rnn`)

Maps to `onnx::RNN`. Weight layout conversion:

- PyTorch: `weight_ih_l0 [H, I]`, `weight_hh_l0 [H, H]`, two bias vectors `[H]` each
- ONNX:    `W [num_dir, H, I]`, `R [num_dir, H, H]`, `B [num_dir, 2H]` (biases concatenated)

Supports `tanh` and `relu` nonlinearities, and bidirectional models.

#### 5. LSTM  (`parse_lstm`)

Maps to `onnx::LSTM`.

PyTorch gate order is **IFGO** (input, forget, cell, output).
ONNX gate order is **IOFC** (input, output, forget, cell).

The parser reorders weight rows before export: `[i, f, g, o] → [i, o, f, g]`.

ONNX weight layout: `W [num_dir, 4H, I]`, `R [num_dir, 4H, H]`, `B [num_dir, 8H]`.
Outputs: `Y`, `Y_h` (final hidden state), `Y_c` (final cell state).
Supports bidirectional.

#### 6. GRU  (`parse_gru`)

Maps to `onnx::GRU`.

PyTorch gate order is **RZN** (reset, update, new).
ONNX gate order is **ZRH** (update, reset, hidden).

The parser reorders: `[r, z, n] → [z, r, n]`.

ONNX weight layout: `W [num_dir, 3H, I]`, `R [num_dir, 3H, H]`, `B [num_dir, 6H]`.
`linear_before_reset=1` is set because PyTorch applies the reset gate after the linear
transform on the hidden state, which is what ONNX `linear_before_reset=1` specifies.
Supports bidirectional.

### Node-info dictionary format

Every `parse_*` function returns a dict matching the C++ parser's internal `fNode` structure:

```python
{
    'nodeType'       : str,                    # ONNX op name, e.g. 'onnx::LSTM'
    'nodeAttributes' : dict,                   # op-specific attributes
    'nodeInputs'     : list[str],              # tensor names: [data, weight, ...]
    'nodeOutputs'    : list[str],              # output tensor names
    'nodeDType'      : list[str],              # dtype per output, e.g. ['Float']
    'weights'        : dict[str, np.ndarray],  # extracted weight tensors
}
```

Weight arrays are in ONNX shape and gate order, so they can be added directly as
initialized tensors via `rmodel.AddInitializedTensor` during integration.

---

## Exercise 5 (Bonus) – Enhancing the Keras Parser

### Files modified / created

| File | Change |
|---|---|
| `bindings/pyroot/pythonizations/python/ROOT/_pythonization/_tmva/_sofie/_parser/_keras/parser.py` | 5 edits – see below |
| `bindings/pyroot/pythonizations/python/ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/conv_transpose.py` | New file |
| `tmva/sofie/test/test_keras_layer_parser.py` | New test file |

### Running

```bash
source /media/harsh/MyPassport1/root-venv/bin/activate
cd /media/harsh/MyPassport1/root/tmva/sofie/test
python3 test_keras_layer_parser.py
```

### Changes to `parser.py`

**1. Imports added**

```python
from .layers.rnn import MakeKerasRNN
from .layers.conv_transpose import MakeKerasConvTranspose
```

**2. LSTM and GRU enabled in `mapKerasLayer`**

These entries existed in the original code but were commented out. They are now active:

```python
"GRU":  MakeKerasRNN,
"LSTM": MakeKerasRNN,
```

**3. Conv2DTranspose added to `mapKerasLayerWithActivation`**

```python
"Conv2DTranspose": MakeKerasConvTranspose,
```

**4. Channels-last transpose extended to Conv2DTranspose**

The transpose-before/after logic for NHWC→NCHW conversion already handled `"Conv2D"`.
All relevant `if fLayerType == "Conv2D"` guards were widened to
`if fLayerType in ("Conv2D", "Conv2DTranspose")`.

**5. GRU bias shape fixed**

Keras stores GRU bias as `[2, 3H]` when `reset_after=True` (the default) and `[3H]`
when `reset_after=False`. The original code path would have passed this to ROOT with
the wrong shape. A dedicated code block now handles GRU biases before the generic path:

```python
if "gru" in fWeightName and "bias" in fWeightName:
    fData = fWeightArray.flatten()              # gives [6H] or [3H]
    if fWeightArray.ndim == 1:                  # reset_after=False
        fData = np.concatenate([fData, np.zeros_like(fData)])
    fWeightTensorShape = [1, int(fData.shape[0])]  # always [1, 6H]
    rmodel.AddInitializedTensor["float"](fWeightName, fWeightTensorShape, fData)
    continue
```

### `layers/conv_transpose.py` – new file

`MakeKerasConvTranspose` creates an `ROperator_ConvTranspose` for `Conv2DTranspose` layers.

Keras kernel shape `[kH, kW, C_out, C_in]` is transposed to ONNX shape `[C_in, C_out, kH, kW]`
via `(3, 2, 0, 1)` in the weight-extraction loop in `parser.py`
(`"conv" in name and ndims == 4` condition already handles this).

Padding mapping:

| Keras | ONNX `auto_pad` |
|---|---|
| `'valid'` | `'VALID'` |
| `'same'`  | `'SAME_UPPER'` |

### LSTM gate reordering

Keras LSTM gate order is **IFCO** (input, forget, cell, output).
ONNX gate order is **IOFC**.

Kernels are stored as `[input, 4H]` and biases as `[4H]` in Keras. The swap is done
in-place on `fWeightArray` before adding the tensor to the RModel:

```
Keras IFCO: [i | f | c | o]  positions 0,1,2,3
ONNX  IOFC: [i | o | f | c]  swap slots 1,2,3: f->2, c->3, o->1
```

### GRU gate reordering

Keras GRU kernel gate order is already **ZRH**, which matches ONNX. No reordering needed
for the kernels. Only the bias shape needs fixing (handled in the GRU bias block above).

---

## Testing Methodology

Tests were designed to be independent of ROOT: no ROOT installation is required to run them.
This makes them easier to run in CI and on any machine with PyTorch and Keras.

### Exercise 4 – `test_pytorch_layer_parser.py`

Structural tests (shapes, attributes, tensor names) plus gate-reordering correctness:

- **ELU**: `alpha` attribute, no weights.
- **MaxPool2D**: `kernel_shape`, `strides`, ONNX-format `pads`, asymmetric case.
- **BatchNorm2D**: all 4 weight arrays extracted, values match the module.
- **RNN**: W/R/B shapes, bidirectional variant, `tanh`/`relu` activation strings.
- **LSTM**: W/R/B shapes; gate reorder verified by constructing the expected IOFC matrix
  from raw PyTorch weights and comparing with `assert_allclose`.
- **GRU**: same gate-reorder check (RZN → ZRH); `linear_before_reset=1` in attributes.
- **`parse_model()` integration**: `MixedModel` (Conv2d + BatchNorm + ELU + MaxPool),
  `RecurrentModel` (LSTM + GRU) – verifies the dispatch map and tensor-name chaining.

### Exercise 5 – `test_keras_layer_parser.py`

**LSTM (Test 1)**

- Known-weight marker test: bias set to `[1,1,..., 2,2,..., 3,3,..., 4,4,...]` (IFCO groups).
  After extraction the IOFC layout must give `[1,1,..., 4,4,..., 2,2,..., 3,3,...]`.
- Numerical forward pass: random weights set on the Keras layer; manual LSTM cell loop
  using the extracted ONNX-order matrices; output compared to Keras with `atol=1e-5`.

**GRU (Test 2)**

- Shape check: `[2, 3H]` Keras bias flattens to `[1, 6H]` ONNX format.
- Bias content verified: `B = [input_bias | recurrent_bias]`.
- Numerical forward pass with `linear_before_reset=1` semantics
  (`n = tanh(Wh*x + bWh + r*(Rh*h + bRh))`).
- `reset_after=False` variant: `[3H]` bias padded to `[1, 6H]` with zeros.

**Conv2DTranspose (Test 3)**

- Explicit-index test: single known entry at `[kH=1, kW=2, C_out=3, C_in=0] = 7.0` in
  Keras layout; after `(3,2,0,1)` transpose this must appear at `[C_in=0, C_out=3, kH=1, kW=2]`
  in ONNX layout. All other entries verified zero.
- Numerical cross-check with `padding='valid'`: ONNX-format weights loaded into
  `torch.nn.ConvTranspose2d`; PyTorch output (after NCHW→NHWC conversion) compared
  to Keras output with `atol=1e-4`.
- Shape-only check for `padding='same'` (frameworks differ in padding distribution,
  but weight extraction is identical).

---

## Screenshots

### Exercise 4 – PyTorch layer parser tests (all passing)

![Exercise 4 test results](screenshots/ex4_pytorch_tests.png)

### Exercise 4 – `parse_model()` demo output

![Exercise 4 demo](screenshots/ex4_demo.png)

### Exercise 5 – Keras parser tests (all passing)

![Exercise 5 test results](screenshots/ex5_keras_tests.png)

---

## References

- [SOFIE source – `RModelParser_PyTorch.cxx`](tmva/sofie_parsers/src/RModelParser_PyTorch.cxx)
- [SOFIE source – Keras parser (`parser.py`)](bindings/pyroot/pythonizations/python/ROOT/_pythonization/_tmva/_sofie/_parser/_keras/parser.py)
- [ONNX operator specs](https://onnx.ai/onnx/operators/)
- [PyTorch LSTM docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch GRU docs](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
