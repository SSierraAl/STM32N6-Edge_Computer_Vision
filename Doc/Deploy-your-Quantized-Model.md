# Quantized model integration

Model deployment combines three components, the generated network code, the corresponding external constants, and application configuration compatible with the exported tensors. The NUCLEO project uses a YOLOv8 instance segmentation adapter with signed detection and prototype outputs.

## Compilation and generated artifacts

ST Edge AI Core compiles the quantized TFLite graph using the NUCLEO profile and memory pool. The `--input-data-type uint8` option requests an input interface compatible with the camera's RGB bytes. The generated scale and zero point establish whether those bytes represent the model's expected normalized values directly.

The generation procedure copies `network.c`, `network_ecblobs.h`, `stai_network.c`, and `stai_network.h` into `Model/NUCLEO-N657X0-Q/`. The external constant buffer is converted to Intel HEX at `0x70380000`. All files must originate from the same generation. [Generation script](../Model/generate-n6-model_NUCLEO-N657X0-Q.sh) · [Compilation notebook](../Notebooks/02_Compile_and_Deploy_NUCLEO_N6.ipynb)

## Tensor compatibility

The reference interface consists of one unsigned 256 × 256 RGB input and two signed outputs. The detection output stores box coordinates, class scores, and mask coefficients in attribute-first order. The second output stores mask prototypes with channels last.

A replacement model requires verification of shape, dtype, quantization, output order, and value semantics. Input dimensions are derived from the generated header, while the postprocessor's candidate counts, prototype dimensions, and quantization pairs are specified in the application configuration. [Generated interface](../Model/NUCLEO-N657X0-Q/stai_network.h#L303) · [Postprocessing definitions](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L131)

## Class and decoder configuration

The class count and table correspond to the label IDs used during training. For a one-class insect dataset, the existing definitions become the following.

```c
#define NB_CLASSES (1)
#define CLASSES_TABLE const char* classes_table[NB_CLASSES] = {"insect"};
```

The current application selects `POSTPROCESS_ISEG_YOLO_V8_UI`. Its adapter reads the detection and prototype scale/zero-point pairs from `app_config.h`. These values require synchronization with the newly generated header. Confidence and NMS thresholds can then be adjusted based on validation results. [Application configuration](../Application/NUCLEO-N657X0-Q/Inc/app_config.h) · [Decoder adapter](../Middlewares/ai-postprocessing-wrapper/app_postprocess_iseg_yolo_v8_ui.c)

## Build, programming, and validation

After interface verification, the application is rebuilt with the generated C files and programmed together with the matching model HEX. A fresh installation also requires the first-stage bootloader. Positive and empty scenes provide initial checks of detection and mask behavior. Numerical comparison with host inference requires identical RGB inputs and examination of the raw output tensors before decoding.

The [README](../README.md#5-build-the-application) provides the build and programming commands. The [source reference](Source-Walkthrough.md) identifies the relevant functions and parameters.
