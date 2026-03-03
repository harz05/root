# GSoC 2026 - ML Inference on Heterogeneous Architectures using SOFIE

**Candidate:** Harsh Chauhan  
**Mentors:** Lorenzo Moneta, Sanjiban Sengupta  
**Branch:** sofie-alpaka-gsoc26  

---

## Exercise 1 - Build ROOT with TMVA-SOFIE

Built ROOT from source with the following options enabled:
- tmva-sofie
- mathmore
- roofit

Build completed successfully. ROOT version confirmed via `root --version`.

---

## Exercise 2 - Run TMVA and SOFIE Tutorials

All tutorials run from `tutorials/machine_learning/`:

**TMVA_Higgs_Classification**
- Best classifier: DNN_CPU with ROC AUC = 0.767
- Other classifiers: BDT 0.758, Likelihood 0.700, Fisher 0.654
- Output: `tmva/sofie/tutorials/Higgs_ClassificationOutput.root`

**TMVA_CNN_Classification**
- Best classifier: CNN_CPU with ROC AUC = 0.770
- Other classifiers: BDT 0.691, DNN 0.668
- Output: `tmva/sofie/tutorials/TMVA_CNN_ClassificationOutput.root`

**TMVA_SOFIE_ONNX**
- Parsed `Linear_16.onnx`, generated C++ inference header via SOFIE
- SOFIE inference output verified against ONNXRuntime output

**TMVA_SOFIE_PyTorch**
- Parsed `PyTorchModel.pt`, generated C++ inference header

**TMVA_SOFIE_Keras - Skipped**
- No `.C` version of this tutorial exists in `tutorials/machine_learning/`
- The `.py` version requires TensorFlow < 2.16 which conflicts with the ROOT pymva installation

---

## Exercise 3 - Explore the SOFIE Alpaka Implementation

The experimental alpaka GPU inference implementation lives in the standalone repo:
https://github.com/ML4EP/SOFIE/tree/gpu/alpaka

The architecture works as follows:
- SOFIE parses ONNX/PyTorch models and generates C++ inference code
- For GPU inference, each operator generates three things:
  - A kernel struct (`Generate_GPU_Kernel_ALPAKA`) defining the GPU computation
  - A kernel declaration (`Generate_GPU_Kernel_Definitions_ALPAKA`) instantiating it
  - A kernel launch (`Generate_GPU_ALPAKA`) dispatching threads via alpaka
- alpaka abstracts over CUDA/HIP/CPU backends so the same kernel code runs anywhere
- The generated `Session<Backend>` class takes device buffers as input and returns device buffers

Existing alpaka operators at the time of this work: Sigmoid, LeakyRelu.

Potential improvements identified:
- Several common activation operators (Tanh, Elu, Softmax) had no GPU implementation
- ROOT's `tmva/sofie` CMakeLists does not yet wire up alpaka tests - this needs integration
- The Softmax operator requires a reduction (not just elementwise), which is a more interesting parallelization problem

---

## Exercise 4 - Implement ONNX Operators for GPU using Alpaka

### Files Modified

**Operator implementations** (in `tmva/sofie/inc/TMVA/`):
- `ROperator_Tanh.hxx`
- `ROperator_Elu.hxx`
- `ROperator_Softmax.hxx`

**Test file** (in `tmva/sofie/test/`):
- `TestCustomModelsFromONNXForAlpakaCuda.cxx`

### What was implemented

Each operator required adding three methods to the existing class:

**Tanh**

Elementwise operator. Each GPU thread handles one element:
```
out[i] = tanh(in[i])
```
Straightforward to parallelize - one thread per element, no dependencies between outputs.

**Elu**

Elementwise operator with an alpha parameter:
```
out[i] = in[i] >= 0 ? in[i] : alpha * (exp(in[i]) - 1)
```
Alpha is read from the ONNX model at parse time and baked directly into the generated kernel launch code, so no runtime parameter passing overhead.

**Softmax**

Reduction-based operator, more complex than the above two. Softmax must be computed along a specified axis, which requires:
1. Finding the maximum value in each row (for numerical stability)
2. Computing exp(x - max) for each element
3. Summing those values
4. Dividing each element by the sum

The implementation decomposes the input tensor into (numRows, rowSize) along the softmax axis and assigns one thread per row. Each thread handles the full reduction for its row sequentially. This is correct and numerically stable, though a more optimized version could use shared memory for the reduction.

### Unit Tests

Three test cases added to `TestCustomModelsFromONNXForAlpakaCuda.cxx`, matching the same inputs used in the CPU tests in `TestCustomModelsFromONNX.cxx`:

- `SofieAlpakaTest.Tanh` - input size 24, random values
- `SofieAlpakaTest.Elu` - input shape [2,3], mixed positive and negative values
- `SofieAlpakaTest.Softmax1d` - input size 3, values [-1, 0, 1]

Each test allocates a host buffer, copies to device, runs `session.infer()`, copies back, and compares element-wise against reference outputs with tolerance 1e-3.

### Test Results

Tests were run on an NVIDIA T4 GPU via Google Colab using the standalone SOFIE repo
(https://github.com/harz05/SOFIE/tree/gpu/alpaka) since ROOT's CMakeLists does not
yet have alpaka test integration.

```
[==========] Running 8 tests from 1 test suite.
[ RUN      ] SofieAlpakaTest.Linear16
[       OK ] SofieAlpakaTest.Linear16 (443 ms)
[ RUN      ] SofieAlpakaTest.Linear32
[       OK ] SofieAlpakaTest.Linear32 (48 ms)
[ RUN      ] SofieAlpakaTest.Linear64
[       OK ] SofieAlpakaTest.Linear64 (18 ms)
[ RUN      ] SofieAlpakaTest.LinearWithLeakyRelu
[       OK ] SofieAlpakaTest.LinearWithLeakyRelu (417 ms)
[ RUN      ] SofieAlpakaTest.LinearWithSigmoid
[       OK ] SofieAlpakaTest.LinearWithSigmoid (2 ms)
[ RUN      ] SofieAlpakaTest.Tanh
[       OK ] SofieAlpakaTest.Tanh (1 ms)
[ RUN      ] SofieAlpakaTest.Elu
[       OK ] SofieAlpakaTest.Elu (1 ms)
[ RUN      ] SofieAlpakaTest.Softmax1d
[       OK ] SofieAlpakaTest.Softmax1d (1 ms)
[----------] 8 tests from SofieAlpakaTest (935 ms total)
[  PASSED  ] 8 tests.
```

### How to Build and Test

Use the standalone SOFIE repo which has the full alpaka CMake setup:

```bash
git clone -b gpu/alpaka https://github.com/harz05/SOFIE.git
cd SOFIE && mkdir build && cd build
cmake .. -Dtesting=ON -DENABLE_ALPAKA_TESTS=ON -DALPAKA_BACKEND=cuda
cmake --build . -j$(nproc)
cd src/SOFIE_core/test && ./TestCustomModelsFromONNXForAlpakaCuda
```

alpaka and sofieBLAS are fetched automatically via CMake FetchContent. Requires CUDA toolkit.

##### A Google Colab notebook for running the tests on a free T4 GPU is included in the SOFIE fork: `SOFIE_Alpaka_Test.ipynb`

---

## Repository Links

- ROOT fork (operator implementations): https://github.com/harz05/root/tree/sofie-alpaka-gsoc26
- SOFIE standalone fork (build + GPU tests): https://github.com/harz05/SOFIE/tree/gpu/alpaka
