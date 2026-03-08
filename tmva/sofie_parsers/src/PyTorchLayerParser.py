## \file PyTorchLayerParser.py
## \ingroup TMVA_SOFIE
## \brief Python-level parsing functions for PyTorch model layers into SOFIE-compatible representation.
##
## This module extends the SOFIE PyTorch parser by implementing Python-level
## parsing functions for operators not yet covered by the C++ parser:
##   - ELU
##   - MaxPool2D
##   - BatchNorm2D
##   - RNN
##   - LSTM
##   - GRU
##
## Each parsing function extracts layer parameters and weights from a loaded
## PyTorch module and returns a node-info dictionary matching the format used
## internally by TMVA::Experimental::SOFIE::PyTorch::Parse (RModelParser_PyTorch.cxx).
##
## Node dictionary format (mirrors C++ parser's fNode structure):
##   {
##     'nodeType'       : str           -- ONNX-style operator name
##     'nodeAttributes' : dict          -- operator attributes (alpha, kernel_shape, etc.)
##     'nodeInputs'     : list[str]     -- input tensor names (data + weights)
##     'nodeOutputs'    : list[str]     -- output tensor names
##     'nodeDType'      : list[str]     -- data types per output
##     'weights'        : dict[str, np.ndarray]  -- extracted weight tensors
##   }
##
## \author Harsh Chauhan (GSoC 2026 candidate)

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_pair(val) -> List[int]:
    """Ensure a value is a 2-element list (handles int or tuple)."""
    if isinstance(val, (list, tuple)):
        return list(val)
    return [int(val), int(val)]


def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach and convert a PyTorch tensor to a NumPy float32 array."""
    return t.detach().float().numpy()


# ---------------------------------------------------------------------------
# Individual layer parsers
# ---------------------------------------------------------------------------

def parse_elu(module: nn.ELU, input_name: str, output_name: str) -> Dict[str, Any]:
    """Parse a torch.nn.ELU layer.

    ELU(x) = x                  if x > 0
           = alpha * (exp(x)-1)  if x <= 0

    Maps to ONNX operator 'onnx::Elu'.
    No learnable weights – only the scalar attribute 'alpha'.

    Parameters
    ----------
    module      : nn.ELU instance
    input_name  : name of the input tensor in the graph
    output_name : name of the output tensor in the graph

    Returns
    -------
    Node-info dict (see module docstring).
    """
    alpha = float(module.alpha) if hasattr(module, 'alpha') else 1.0

    return {
        'nodeType'       : 'onnx::Elu',
        'nodeAttributes' : {'alpha': alpha},
        'nodeInputs'     : [input_name],
        'nodeOutputs'    : [output_name],
        'nodeDType'      : ['Float'],
        'weights'        : {}          # ELU has no learnable parameters
    }


def parse_maxpool2d(module: nn.MaxPool2d,
                    input_name: str,
                    output_name: str) -> Dict[str, Any]:
    """Parse a torch.nn.MaxPool2d layer.

    Maps to ONNX operator 'onnx::MaxPool'.
    Padding is stored in ONNX format: [pad_top, pad_left, pad_bottom, pad_right].

    Parameters
    ----------
    module      : nn.MaxPool2d instance
    input_name  : name of the input tensor
    output_name : name of the output tensor

    Returns
    -------
    Node-info dict.
    """
    kernel_shape = _to_pair(module.kernel_size)
    strides      = _to_pair(module.stride if module.stride is not None else module.kernel_size)
    pad          = _to_pair(module.padding)
    dilations    = _to_pair(module.dilation)
    # ONNX pads: [x1_begin, x2_begin, x1_end, x2_end]
    pads         = [pad[0], pad[1], pad[0], pad[1]]

    return {
        'nodeType'       : 'onnx::MaxPool',
        'nodeAttributes' : {
            'kernel_shape' : kernel_shape,
            'strides'      : strides,
            'pads'         : pads,
            'dilations'    : dilations,
            'ceil_mode'    : int(module.ceil_mode),
        },
        'nodeInputs'     : [input_name],
        'nodeOutputs'    : [output_name],
        'nodeDType'      : ['Float'],
        'weights'        : {}          # MaxPool2d has no learnable parameters
    }


def parse_batchnorm2d(module: nn.BatchNorm2d,
                      input_name: str,
                      output_name: str,
                      layer_name: str) -> Dict[str, Any]:
    """Parse a torch.nn.BatchNorm2d layer.

    Maps to ONNX operator 'onnx::BatchNormalization'.
    Extracts: scale (gamma), bias (beta), running_mean, running_var.

    Input tensor order expected by ONNX BatchNorm:
      [X, scale, bias, mean, var]

    Parameters
    ----------
    module      : nn.BatchNorm2d instance
    input_name  : name of the activation input tensor
    output_name : name of the output tensor
    layer_name  : unique prefix for naming weight tensors

    Returns
    -------
    Node-info dict with weights dict containing the 4 parameter tensors.
    """
    scale_name  = layer_name + '_scale'
    bias_name   = layer_name + '_bias'
    mean_name   = layer_name + '_running_mean'
    var_name    = layer_name + '_running_var'

    weights = {
        scale_name : _tensor_to_numpy(module.weight),
        bias_name  : _tensor_to_numpy(module.bias),
        mean_name  : _tensor_to_numpy(module.running_mean),
        var_name   : _tensor_to_numpy(module.running_var),
    }

    return {
        'nodeType'       : 'onnx::BatchNormalization',
        'nodeAttributes' : {
            'epsilon'       : float(module.eps),
            'momentum'      : float(module.momentum) if module.momentum else 0.1,
            'training_mode' : 0,       # inference mode
        },
        'nodeInputs'     : [input_name, scale_name, bias_name, mean_name, var_name],
        'nodeOutputs'    : [output_name],
        'nodeDType'      : ['Float'],
        'weights'        : weights
    }


def parse_rnn(module: nn.RNN,
              input_name: str,
              output_name: str,
              layer_name: str) -> Dict[str, Any]:
    """Parse a torch.nn.RNN layer.

    Maps to ONNX operator 'onnx::RNN'.

    PyTorch RNN weight layout (single direction, single layer):
      weight_ih_l0 : [hidden_size, input_size]
      weight_hh_l0 : [hidden_size, hidden_size]
      bias_ih_l0   : [hidden_size]
      bias_hh_l0   : [hidden_size]

    ONNX RNN weight layout:
      W : [num_directions, hidden_size, input_size]
      R : [num_directions, hidden_size, hidden_size]
      B : [num_directions, 2 * hidden_size]   (bias_ih concat bias_hh)

    Parameters
    ----------
    module      : nn.RNN instance
    input_name  : name of the input tensor  (shape [seq_len, batch, input_size])
    output_name : name of the output tensor
    layer_name  : unique prefix for weight tensor names

    Returns
    -------
    Node-info dict with W, R, B weight tensors.
    """
    num_directions = 2 if module.bidirectional else 1

    W_list, R_list, B_list = [], [], []
    for d in range(num_directions):
        suffix = '_reverse' if d == 1 else ''
        W_list.append(_tensor_to_numpy(getattr(module, f'weight_ih_l0{suffix}')))
        R_list.append(_tensor_to_numpy(getattr(module, f'weight_hh_l0{suffix}')))
        if module.bias:
            b_ih = _tensor_to_numpy(getattr(module, f'bias_ih_l0{suffix}'))
            b_hh = _tensor_to_numpy(getattr(module, f'bias_hh_l0{suffix}'))
            B_list.append(np.concatenate([b_ih, b_hh]))
        else:
            B_list.append(np.zeros(2 * module.hidden_size, dtype=np.float32))

    W_name = layer_name + '_W'
    R_name = layer_name + '_R'
    B_name = layer_name + '_B'

    weights = {
        W_name : np.stack(W_list, axis=0),   # [num_dir, hidden, input]
        R_name : np.stack(R_list, axis=0),   # [num_dir, hidden, hidden]
        B_name : np.stack(B_list, axis=0),   # [num_dir, 2*hidden]
    }

    return {
        'nodeType'       : 'onnx::RNN',
        'nodeAttributes' : {
            'hidden_size'       : module.hidden_size,
            'activations'       : [module.nonlinearity.upper()],  # 'TANH' or 'RELU'
            'direction'         : 'bidirectional' if module.bidirectional else 'forward',
        },
        'nodeInputs'     : [input_name, W_name, R_name, B_name],
        'nodeOutputs'    : [output_name],
        'nodeDType'      : ['Float'],
        'weights'        : weights
    }


def parse_lstm(module: nn.LSTM,
               input_name: str,
               output_name: str,
               layer_name: str) -> Dict[str, Any]:
    """Parse a torch.nn.LSTM layer.

    Maps to ONNX operator 'onnx::LSTM'.

    PyTorch LSTM gates order: i (input), f (forget), g (cell), o (output) -- IFGO
    ONNX    LSTM gates order: i (input), o (output), f (forget), c (cell) -- IOFC

    So we must reorder weight rows: PyTorch [i,f,g,o] -> ONNX [i,o,f,g]

    ONNX weight layout:
      W : [num_directions, 4*hidden_size, input_size]
      R : [num_directions, 4*hidden_size, hidden_size]
      B : [num_directions, 8*hidden_size]

    Parameters
    ----------
    module      : nn.LSTM instance
    input_name  : name of input tensor (shape [seq_len, batch, input_size])
    output_name : base name for output tensor
    layer_name  : unique prefix for weight tensor names

    Returns
    -------
    Node-info dict with properly reordered W, R, B tensors.
    """
    num_directions = 2 if module.bidirectional else 1
    H = module.hidden_size

    def _reorder_lstm_gates(tensor: np.ndarray) -> np.ndarray:
        """Reorder gate axis from PyTorch IFGO to ONNX IOFC."""
        # tensor shape: [4*H, ...] -- split into 4 gates along axis 0
        i, f, g, o = np.split(tensor, 4, axis=0)
        return np.concatenate([i, o, f, g], axis=0)

    W_list, R_list, B_list = [], [], []
    for d in range(num_directions):
        suffix = '_reverse' if d == 1 else ''
        w_ih = _tensor_to_numpy(getattr(module, f'weight_ih_l0{suffix}'))
        w_hh = _tensor_to_numpy(getattr(module, f'weight_hh_l0{suffix}'))
        W_list.append(_reorder_lstm_gates(w_ih))
        R_list.append(_reorder_lstm_gates(w_hh))
        if module.bias:
            b_ih = _tensor_to_numpy(getattr(module, f'bias_ih_l0{suffix}'))
            b_hh = _tensor_to_numpy(getattr(module, f'bias_hh_l0{suffix}'))
            B_list.append(np.concatenate([
                _reorder_lstm_gates(b_ih),
                _reorder_lstm_gates(b_hh)
            ]))
        else:
            B_list.append(np.zeros(8 * H, dtype=np.float32))

    W_name = layer_name + '_W'
    R_name = layer_name + '_R'
    B_name = layer_name + '_B'

    weights = {
        W_name : np.stack(W_list, axis=0),   # [num_dir, 4*H, input_size]
        R_name : np.stack(R_list, axis=0),   # [num_dir, 4*H, H]
        B_name : np.stack(B_list, axis=0),   # [num_dir, 8*H]
    }

    return {
        'nodeType'       : 'onnx::LSTM',
        'nodeAttributes' : {
            'hidden_size' : H,
            'direction'   : 'bidirectional' if module.bidirectional else 'forward',
        },
        'nodeInputs'     : [input_name, W_name, R_name, B_name],
        'nodeOutputs'    : [output_name + '_Y', output_name + '_Y_h', output_name + '_Y_c'],
        'nodeDType'      : ['Float', 'Float', 'Float'],
        'weights'        : weights
    }


def parse_gru(module: nn.GRU,
              input_name: str,
              output_name: str,
              layer_name: str) -> Dict[str, Any]:
    """Parse a torch.nn.GRU layer.

    Maps to ONNX operator 'onnx::GRU'.

    PyTorch GRU gates order: r (reset), z (update), n (new/hidden) -- RZN
    ONNX    GRU gates order: z (update), r (reset), h (hidden)     -- ZRH

    So we must reorder: PyTorch [r, z, n] -> ONNX [z, r, n]

    ONNX weight layout:
      W : [num_directions, 3*hidden_size, input_size]
      R : [num_directions, 3*hidden_size, hidden_size]
      B : [num_directions, 6*hidden_size]

    Parameters
    ----------
    module      : nn.GRU instance
    input_name  : name of input tensor
    output_name : name of output tensor
    layer_name  : unique prefix for weight tensor names

    Returns
    -------
    Node-info dict with properly reordered W, R, B tensors.
    """
    num_directions = 2 if module.bidirectional else 1
    H = module.hidden_size

    def _reorder_gru_gates(tensor: np.ndarray) -> np.ndarray:
        """Reorder gate axis from PyTorch RZN to ONNX ZRH."""
        r, z, n = np.split(tensor, 3, axis=0)
        return np.concatenate([z, r, n], axis=0)

    W_list, R_list, B_list = [], [], []
    for d in range(num_directions):
        suffix = '_reverse' if d == 1 else ''
        w_ih = _tensor_to_numpy(getattr(module, f'weight_ih_l0{suffix}'))
        w_hh = _tensor_to_numpy(getattr(module, f'weight_hh_l0{suffix}'))
        W_list.append(_reorder_gru_gates(w_ih))
        R_list.append(_reorder_gru_gates(w_hh))
        if module.bias:
            b_ih = _tensor_to_numpy(getattr(module, f'bias_ih_l0{suffix}'))
            b_hh = _tensor_to_numpy(getattr(module, f'bias_hh_l0{suffix}'))
            B_list.append(np.concatenate([
                _reorder_gru_gates(b_ih),
                _reorder_gru_gates(b_hh)
            ]))
        else:
            B_list.append(np.zeros(6 * H, dtype=np.float32))

    W_name = layer_name + '_W'
    R_name = layer_name + '_R'
    B_name = layer_name + '_B'

    weights = {
        W_name : np.stack(W_list, axis=0),   # [num_dir, 3*H, input_size]
        R_name : np.stack(R_list, axis=0),   # [num_dir, 3*H, H]
        B_name : np.stack(B_list, axis=0),   # [num_dir, 6*H]
    }

    return {
        'nodeType'       : 'onnx::GRU',
        'nodeAttributes' : {
            'hidden_size'         : H,
            'direction'           : 'bidirectional' if module.bidirectional else 'forward',
            'linear_before_reset' : 1,  # PyTorch GRU applies r AFTER linear(h), matching ONNX linear_before_reset=1
        },
        'nodeInputs'     : [input_name, W_name, R_name, B_name],
        'nodeOutputs'    : [output_name + '_Y', output_name + '_Y_h'],
        'nodeDType'      : ['Float', 'Float'],
        'weights'        : weights
    }


# ---------------------------------------------------------------------------
# Dispatch map (mirrors mapPyTorchNode in RModelParser_PyTorch.cxx)
# ---------------------------------------------------------------------------

LAYER_PARSER_MAP = {
    nn.ELU          : parse_elu,
    nn.MaxPool2d    : parse_maxpool2d,
    nn.BatchNorm2d  : parse_batchnorm2d,
    nn.RNN          : parse_rnn,
    nn.LSTM         : parse_lstm,
    nn.GRU          : parse_gru,
}


# ---------------------------------------------------------------------------
# Top-level model parser
# ---------------------------------------------------------------------------

def parse_model(model: nn.Module,
                input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """Parse a PyTorch model, extracting all supported new layer types.

    Iterates over the model's named modules and calls the appropriate
    parse_* function for each supported layer type.  Layers not yet
    supported (e.g. nn.Linear, nn.ReLU – already handled by the C++
    parser) are catalogued but not parsed here.

    Parameters
    ----------
    model       : any nn.Module (must be in eval() mode)
    input_shape : shape of a single input sample (without batch dim)

    Returns
    -------
    dict with keys:
      'nodes'           : list of node-info dicts (one per supported layer)
      'weights'         : flat dict of all extracted weight tensors
      'unsupported'     : list of (layer_name, type_str) for unsupported layers
      'input_shape'     : passed-through input_shape
    """
    model.eval()

    all_nodes    : List[Dict[str, Any]] = []
    all_weights  : Dict[str, np.ndarray] = {}
    unsupported  : List[Tuple[str, str]] = []

    # simple counter to generate unique tensor names
    tensor_counter = [0]

    def _next_tensor(prefix: str) -> str:
        name = f"{prefix}_{tensor_counter[0]}"
        tensor_counter[0] += 1
        return name

    prev_output = "input"

    for layer_name, module in model.named_modules():
        if module is model:
            continue   # skip the top-level module itself

        module_type = type(module)
        out_name    = _next_tensor(layer_name.replace('.', '_') or 'tensor')

        if module_type in LAYER_PARSER_MAP:
            parser = LAYER_PARSER_MAP[module_type]

            # BatchNorm2D, RNN, LSTM, GRU need the layer_name for weight naming
            needs_layer_name = module_type in (nn.BatchNorm2d, nn.RNN, nn.LSTM, nn.GRU)

            if needs_layer_name:
                node = parser(module, prev_output, out_name,
                              layer_name.replace('.', '_'))
            else:
                node = parser(module, prev_output, out_name)

            all_nodes.append(node)
            all_weights.update(node['weights'])
            prev_output = node['nodeOutputs'][0]   # chain: output -> next input
        else:
            unsupported.append((layer_name, module_type.__name__))
            # Still advance the tensor name so chaining isn't broken
            prev_output = out_name

    return {
        'nodes'        : all_nodes,
        'weights'      : all_weights,
        'unsupported'  : unsupported,
        'input_shape'  : input_shape,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_parsed_model(parsed: Dict[str, Any]) -> None:
    """Print a human-readable summary of a parsed model."""
    print("=" * 60)
    print("SOFIE PyTorch Parser - Parsed Layer Summary")
    print("=" * 60)
    print(f"Input shape : {parsed['input_shape']}")
    print(f"Nodes found : {len(parsed['nodes'])}")
    print()

    for i, node in enumerate(parsed['nodes']):
        print(f"  [{i}] {node['nodeType']}")
        print(f"       inputs  : {node['nodeInputs']}")
        print(f"       outputs : {node['nodeOutputs']}")
        if node['nodeAttributes']:
            print(f"       attrs   : {node['nodeAttributes']}")
        if node['weights']:
            for wname, wdata in node['weights'].items():
                print(f"       weight  : {wname}  shape={wdata.shape}  dtype={wdata.dtype}")
        print()

    if parsed['unsupported']:
        print("Unsupported layers (handled by C++ parser or not yet implemented):")
        for lname, ltype in parsed['unsupported']:
            print(f"  {lname!r:30s} -> {ltype}")
    print("=" * 60)
