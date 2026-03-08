import math

from .. import get_keras_version


def MakeKerasConvTranspose(layer):
    """
    Create a Keras-compatible transposed convolution layer operation using SOFIE framework.

    This function takes a dictionary representing a Conv2DTranspose layer and its attributes
    and constructs the corresponding SOFIE ROperator_ConvTranspose operation.

    A transposed convolution (also called deconvolution or fractionally-strided convolution)
    is the gradient of a standard convolution with respect to its input. It is commonly used
    in generative models and semantic segmentation architectures to upsample feature maps.

    Keras Conv2DTranspose kernel shape:  [kH, kW, C_out, C_in]  (channels_last default)
    ONNX ConvTranspose weight shape:     [C_in, C_out/group, kH, kW]
    The transpose (3,2,0,1) applied in Parse() converts Keras -> ONNX format automatically.

    Padding mapping:
      Keras 'valid' -> ONNX auto_pad='VALID'   (no padding; output grows)
      Keras 'same'  -> ONNX auto_pad='SAME_UPPER'
                       (output = input_spatial * stride; padding added symmetrically)

    Parameters
    ----------
    layer : dict
        Layer-info dictionary produced by PyKeras.Parse, containing:
          layerInput, layerOutput, layerDType, layerAttributes, layerWeight.

    Returns
    -------
    ROperator_ConvTranspose: SOFIE operator for the transposed convolution.

    Raises
    ------
    RuntimeError  If dtype is not float or padding mode is unsupported.
    """
    from ROOT.TMVA.Experimental import SOFIE

    keras_version = get_keras_version()

    finput = layer["layerInput"]
    foutput = layer["layerOutput"]
    fLayerDType = layer["layerDType"]
    fLayerInputName = finput[0]
    fLayerOutputName = foutput[0]
    attributes = layer["layerAttributes"]
    fWeightNames = layer["layerWeight"]
    fKernelName = fWeightNames[0]
    fBiasName = fWeightNames[1] if len(fWeightNames) > 1 else ""

    # Core spatial attributes
    fAttrDilations = list(attributes["dilation_rate"])
    fAttrGroup = int(attributes.get("groups", 1))
    fAttrKernelShape = list(attributes["kernel_size"])
    fAttrStrides = list(attributes["strides"])
    fKerasPadding = str(attributes["padding"])

    # output_padding / output_shape: empty unless explicitly set
    fAttrOutputPadding = []
    fAttrOutputShape = []
    fAttrPads = []

    # Padding conversion: Keras -> ONNX
    # 'valid'  -> no padding, output expands: H_out = (H_in-1)*s + k
    # 'same'   -> output kept at H_in*s;  SAME_UPPER pads symmetrically
    if fKerasPadding == "valid":
        fAttrAutopad = "VALID"
    elif fKerasPadding == "same":
        fAttrAutopad = "SAME_UPPER"
    else:
        raise RuntimeError(
            "TMVA::SOFIE - RModel Keras Parser doesn't yet support "
            "Conv2DTranspose with padding '" + fKerasPadding + "'"
        )

    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_ConvTranspose["float"](
            fAttrAutopad,
            fAttrDilations,
            fAttrGroup,
            fAttrKernelShape,
            fAttrOutputPadding,
            fAttrOutputShape,
            fAttrPads,
            fAttrStrides,
            fLayerInputName,
            fKernelName,
            fBiasName,
            fLayerOutputName,
        )
        return op
    else:
        raise RuntimeError(
            "TMVA::SOFIE - Unsupported - Operator ConvTranspose does not yet "
            "support input type " + fLayerDType
        )
