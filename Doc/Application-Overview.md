# STM32N6 computer vision architecture

The application implements real-time instance segmentation on the NUCLEO-N657X0-Q using a compiled YOLOv8n-seg network with a 256 × 256 RGB input. Its architecture comprises model preparation on the workstation, graph compilation for Neural-ART, and firmware execution from camera capture to USB visualization. The following sections describe these stages and their shared tensor and memory interfaces. [Build and deployment procedures](../README.md) · [Source and parameter reference](Source-Walkthrough.md)

## 1. System components

| Component | Responsibility | Implementation reference |
| --- | --- | --- |
| Python training and export | Learns model weights from labeled images. Exports an inference graph and quantizes the values used during inference. | [Notebook 01](../Notebooks/01_Train_and_Quantize_STM32N6.ipynb) |
| ST Edge AI Core on the PC | Compiles the TFLite graph for the Neural-ART accelerator, decides where tensors and constants will live, and emits C code and flash data. | [Generation script](../Model/generate-n6-model_NUCLEO-N657X0-Q.sh) and [Notebook 02](../Notebooks/02_Compile_and_Deploy_NUCLEO_N6.ipynb) |
| Firmware on the STM32N6 | Captures images, invokes the compiled network, decodes the tensors, and draws the results. The Cortex-M55 runs the application and postprocessing. The NPU accelerates supported graph operations. | [`main.c`](../Application/NUCLEO-N657X0-Q/Src/main.c) |

An **inference** is one forward pass from an input image to output tensors using fixed weights. A **tensor** is a typed array whose shape and layout describe how to read its values. **Postprocessing** turns those arrays into objects with boxes, class IDs, scores, and masks.

## 2. Acquisition and inference pipeline

```mermaid
flowchart TD
    Sensor["Camera sensor"] --> ISP["CSI and DCMIPP image pipeline"]
    ISP --> Display["Pipe 1: continuous RGB565"]
    ISP --> Snapshot["Pipe 2: RGB888 snapshot"]
    Snapshot --> Network["Neural-ART inference"]
    Network --> Decode["CPU boxes, classes, masks"]
    Display --> Blend["Foreground and background composition"]
    Decode --> Blend
    Blend --> USB["USB camera stream"]
```

### DCMIPP and ISP

The sensor supplies camera data over CSI. DCMIPP is the STM32N6 camera pipeline. Its image signal processor (ISP) and downstream pipes prepare images for the application. The camera middleware configures the sensor and pipeline, and `CameraPipeline_IspUpdate()` calls `CMW_CAMERA_Run()` to service the ISP. The two outputs share the camera source and can use different formats and dimensions. [STM32N6 reference manual, RM0486](https://www.st.com/resource/en/reference_manual/rm0486-stm32n647657xx-armbased-32bit-mcus-stmicroelectronics.pdf)

| Setting | Pipe 1 | Pipe 2 |
| --- | --- | --- |
| Purpose | Background video for display | Input tensor for inference |
| Capture mode | Continuous | One snapshot per inference loop |
| Pixel format in this build | RGB565, 2 bytes per pixel | RGB888, 3 bytes per pixel |
| Output size | Computed for the display | From the generated `STAI_NETWORK_IN_1_*` macros |
| Destination | `lcd_bg_buffer` | `nn_in` directly when pitch matches, or a temporary buffer otherwise |

**Pitch** is the number of bytes from one row start to the next. At 256 pixels wide, a densely packed RGB888 row takes `256 × 3 = 768` bytes. The whole input takes `256 × 768 = 196,608` bytes, or 192 KiB. If a different width produces a larger hardware pitch, the application captures to `dcmipp_out_nn` and calls `img_crop()` to copy the valid pixels into tightly packed rows. That step deals with row padding. Image framing and resizing are set in `app_camerapipeline.c`.

The default `ASPECT_RATIO_FULLSCREEN` selects the fullscreen path for display and the FIT path for inference. The inference path resizes the camera field into a square, so an object can change aspect ratio. Numerical comparison between PC and board requires identical RGB bytes. Desktop resizing supports visual assessment, while ISP and resampling differences can alter individual pixel values.

### Execution sequence and observable state

| Order | Function | Relevant state or operation |
| --- | --- | --- |
| 1 | `Hardware_init()` | Clocks, UART, NPU RAM and cache, external NOR in memory-mapped mode, peripheral access attributes. |
| 2 | `NeuralNetwork_init()` | `stai_runtime_init()` initializes the runtime. `stai_network_init()` initializes the network instance. The getters provide the actual input and output addresses and sizes. |
| 3 | `CameraPipeline_Init()` | Generated input dimensions, negotiated camera dimensions, and `pitch_nn` establish the capture layout. |
| 4 | `CameraPipeline_NNPipe_Start()` and `CMW_CAMERA_PIPE_FrameEventCallback()` | The foreground loop starts a snapshot. The Pipe 2 frame callback increments `cameraFrameReceived`. The input buffer is available for inspection after capture completion. |
| 5 | `Run_Inference()` | `stai_network_run(..., STAI_MODE_ASYNC)` advances inference. If the runtime requests it, the CPU waits for an event and re-enters the runtime. The main loop waits for this call to finish before decoding. |
| 6 | `app_postprocess_run()` | Raw detection and prototype buffers enter the adapter, which returns decoded objects. |
| 7 | `Display_NetworkOutput()` | Converts normalized boxes to display coordinates, draws into a double-buffered foreground, cleans the cache for the written buffer, and reloads the layer. |

The CPU, camera, display, and NPU access shared memory. A **cache clean** writes dirty CPU cache lines back to RAM. An **invalidate** discards cached lines so subsequent CPU reads can see updated RAM. The required cache operation depends on the buffer's producer, consumer, and ownership transition. Buffer addresses, pitch, and cache-line alignment matter independently.

The main loop and camera implementation define the capture-to-inference handover. [Main loop](../Application/NUCLEO-N657X0-Q/Src/main.c#L161) · [Camera implementation](../Application/NUCLEO-N657X0-Q/Src/app_camerapipeline.c) The UART console uses 115200 baud, 8N1. The displayed inference timer surrounds `Run_Inference()`. It excludes camera wait, postprocessing, and drawing. A complete frame-latency measurement also includes capture, postprocessing, and display operations.

## 3. Tensor representation and quantization

**Input.** One image, 256 rows, 256 columns, and 3 channels, stored as NHWC `[1, 256, 256, 3]`. The camera supplies interleaved RGB bytes. The TFLite host model uses signed `int8` with scale about 1/255 and zero point −128. The generated firmware interface uses `uint8` with scale about 1/255 and zero point 0.

**Worked example.** An RGB channel byte of 128 represents approximately `128 / 255 = 0.502`. The host TFLite receives signed value `0` and reconstructs `(0 - (-128)) / 255`. The generated firmware interface receives unsigned value `128` and reconstructs `(128 - 0) / 255`. For a replacement model, the generated scale and zero point determine compatibility with raw camera bytes.

| Output | COCO TFLite shape | Firmware interpretation |
| --- | --- | --- |
| Candidate detections | `[1, 116, 1344]` | 116 attribute rows by 1,344 candidates in a flat buffer. The generated ST header reports this buffer as `[116, 1344, 1]`. |
| Mask prototypes | `[1, 64, 64, 32]` | A shared 64 × 64 grid of 32 prototype values per pixel with channels last. |

For this graph, the detection head covers grids of 32 × 32, 16 × 16, and 8 × 8, for a total of 1,344 candidates. Each candidate has four box values, 80 class scores, and 32 mask coefficients. Using one class with the same head changes the attribute count to `4 + 1 + 32 = 37`. The contract includes normalized center coordinates and dimensions, class-score semantics, output order, signedness, and memory layout. Matching element counts alone can't establish compatibility.

**Example index.** In the detection buffer, attribute `a` for candidate `i` is at flat offset `a * 1344 + i`. In the prototype buffer, channel `k` at pixel `(x, y)` is at `(y * 64 + x) * 32 + k`. These offsets define the correspondence between Python array indexing and the C buffers.

### Detection filtering and mask reconstruction

1. The decoder selects the highest-scoring class for each candidate and applies the confidence threshold. With COCO's detection scale of approximately 0.02002 and zero point 25, raw score 55 corresponds to about 0.601. A confidence threshold of 0.5 maps to approximately raw value 50 in the bundled decoder.
2. The decoder applies **non-maximum suppression (NMS)** per class. NMS suppresses a lower-scoring box when it overlaps a higher-scoring box above the configured intersection-over-union (IoU) threshold. IoU is intersection area divided by union area. The demo defaults are 0.5 confidence, 0.4 IoU, and at most 10 returned objects.
3. Each retained detection receives a reconstructed binary mask. At each prototype-grid pixel, a dot product combines the candidate's 32 coefficients with the 32 prototype values. The C implementation subtracts the corresponding zero points, accumulates integer products, and compares the sum against a threshold derived from the two scales. The display clips the mask overlay to its corresponding box.
4. The display routine maps normalized box coordinates to display dimensions. For a model-space box centered at `(0.5, 0.5)` with width `0.25`, its left edge is `(0.5 - 0.25/2) × 256 = 96` model pixels. Mapping results to the sensor image requires the inverse preprocessing transform, including any cropping, padding, or mirroring.

[The adapter](../Middlewares/ai-postprocessing-wrapper/app_postprocess_iseg_yolo_v8_ui.c) specifies tensor order and configuration, while [the decoder](../Middlewares/stm32-vision-models-postprocessing/lib_vision_models_pp/Src/iseg_pp_yolov8.c) implements indexing, NMS, and mask reconstruction. The adapter currently reads both output quantization pairs from [`app_config.h`](../Application/NUCLEO-N657X0-Q/Inc/app_config.h). Regenerating the network does not update those definitions.

## 4. Model compilation and deployment artifacts

| Artifact | Contents | Consumer |
| --- | --- | --- |
| `best.pt` | The best checkpoint selected by validation during training | Ultralytics evaluation and export |
| `*_full_integer_quant.tflite` | Portable inference graph, weights, and quantization metadata | PC interpreter and ST Edge AI Core |
| `network.c` and `network_ecblobs.h` | Generated execution code and accelerator data for this graph | Arm firmware build |
| `stai_network.c` and `stai_network.h` | Generated interface, context, and tensor definitions | App, camera pipeline, and ST runtime |
| `network_atonbuf.xSPI2.raw` | Bytes for model constants in the xSPI flash pool | `objcopy` |
| `network_data.hex` | The same model data in Intel HEX with its flash destination encoded | STM32CubeProgrammer |
| `network_generate_report.txt` | Memory allocations, graph details, and compiler execution breakdown | Developer reviewing placement and operator support |

`stedgeai generate` emits the model-specific execution plan and byte data. The Arm build compiles the generated C together with the firmware. `objcopy` packages the external data bytes with address `0x70380000`. CubeProgrammer then writes those records to the board. The firmware calls the generated code and accesses the model data through memory-mapped NOR. The C/H files and model HEX must originate from the same generation.

### Coordination of firmware and network memory

| Allocation input | Managed storage | Verification artifact |
| --- | --- | --- |
| [Firmware linker script](../Application/NUCLEO-N657X0-Q/STM32CubeIDE/STM32N657xx.ld) | Application code, globals, stack and heap, including display and decoding buffers | Build `.map` and linker memory summary |
| [Neural-ART memory pool](../Model/my_mpools/stm32n6-app2_NUCLEO-N657X0-Q.mpool) | Network activation and constant placement for `stedgeai` | [Generation report](../Model/st_ai_output/network_generate_report.txt) |

**Weights** are fixed model parameters. **Activations** are intermediate values created during a forward pass. The current application linker region begins at `0x34000400` and ends before `0x3419C000`. The model pool's 400 KiB CPU-side SRAM region starts at `0x3419C000`. This boundary is an agreement between two tools. Moving it in one file alone can create overlapping allocations. The bundled report shows about 3.314 MiB of constants in external flash and 445 of 448 KiB in one active NPU SRAM bank. Summing free memory across banks can hide placement limits. The used address ranges and software or hybrid epochs provide evidence for investigating memory constraints and latency. [ST Neural-ART compiler primer](https://stedgeai-dc.st.com/assets/embedded-docs/stneuralart_neural_art_compiler.html)

### Boot and deployment

On a flash boot, the boot ROM loads the first-stage bootloader (FSBL). This project's FSBL loads the app for execution in SRAM. The app initializes and maps the NOR holding the network data. Flash locations identify where images are programmed. Linker addresses identify where the application runs.

| Flash start | Contents | When to update |
| --- | --- | --- |
| `0x70000000` | `FSBL/ai_fsbl.hex` | Initial setup or a bootloader change |
| `0x70100000` | Application image with a boot header and generated network code | Firmware or generated network change |
| `0x70380000` | `network_data.hex` | Model regeneration |

During SRAM debugging, the debugger loads the application ELF. The model's external constants still need to be present. The [README flash section](../README.md#6-program-external-flash) gives the actual programming commands, boot jumper positions, and erase behavior.

## 5. Validation procedure

| Validation stage | Procedure | Recorded evidence |
| --- | --- | --- |
| Camera to tensor | A breakpoint after the Pipe 2 snapshot exposes `pitch_nn` and the first few rows of `nn_in`. | Shape, row pitch, channel order, and a known board input. |
| Tensor to objects | The COCO notebook and firmware decoder provide corresponding views of candidate attributes and decoded objects. | Both output shapes, quantization pairs, decoded class and box, and the final mask. For numeric parity, use the same input bytes. |
| One custom model | A one-class export is compared with its PyTorch checkpoint, compiled for STM32, and deployed with matching application and model data. | Validation metrics, generation report, linker map, UART model name, and results on positive and empty captures. |

Errors reproduced by host inference indicate a need to examine labels, training, or quantization. Differences confined to the board motivate verification of RGB order, framing, generated-file consistency, programmed constants, and decoder parameters. Separate timing of acquisition, inference, postprocessing, and display identifies the contributions to frame latency.
