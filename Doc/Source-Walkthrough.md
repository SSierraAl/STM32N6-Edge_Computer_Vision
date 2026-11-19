# C inference source and configuration reference

This reference maps the execution sequence to source functions and configuration definitions. In a VS Code workspace opened at the repository root, **Ctrl+Shift+V** displays Markdown Preview and **Ctrl+K V** opens a preview beside the editor on Windows. Source links use relative paths and `#L<number>` fragments to select the relevant line. Function and macro names provide search references when source changes shift line positions. [VS Code Markdown preview](https://code.visualstudio.com/docs/languages/markdown) · [Source-line links](https://github.com/microsoft/vscode/issues/200709)

C/C++ symbol navigation uses **F12** for definitions, **Alt+F12** for inline inspection, **Shift+F12** for references, and **Alt+Left** for navigation history on Windows. Microsoft's C/C++ extension requires IntelliSense configuration consistent with the NUCLEO compiler, include paths, and build definitions. A compilation database can supply these settings when available. Markdown file links operate independently of symbol resolution. [VS Code code navigation](https://code.visualstudio.com/docs/editing/editingevolved) · [Configure C/C++ IntelliSense](https://code.visualstudio.com/docs/cpp/configure-intellisense)

## 1. Execution sequence

The table follows one acquisition and inference iteration. The [architecture description](Application-Overview.md) provides the corresponding data-flow diagram, tensor equations, and memory allocation.

| Stage | Source location | Operation and relevant state |
| --- | --- | --- |
| 1. Start the app | [`main()`](../Application/NUCLEO-N657X0-Q/Src/main.c#L161) and [`Hardware_init()`](../Application/NUCLEO-N657X0-Q/Src/main.c#L258) | The CPU initializes clocks, UART, NPU RAM and external NOR. The [memory-mapped NOR call](../Application/NUCLEO-N657X0-Q/Src/main.c#L292) makes the flashed weights accessible at their generated addresses. |
| 2. Bind the model and buffers | [`NeuralNetwork_init()`](../Application/NUCLEO-N657X0-Q/Src/main.c#L315) → [generated `stai_network_init()`](../Model/NUCLEO-N657X0-Q/stai_network.c#L359) | The app initializes the runtime and network instance. The relevant variables are `nn_in`, `nn_in_len`, `nn_out[]`, and `nn_out_len[]`. The generated getters supply these buffers. |
| 3. Configure capture | [`CameraPipeline_Init()`](../Application/NUCLEO-N657X0-Q/Src/app_camerapipeline.c#L118) → [Pipe 1 settings](../Application/NUCLEO-N657X0-Q/Src/app_camerapipeline.c#L35) and [Pipe 2 settings](../Application/NUCLEO-N657X0-Q/Src/app_camerapipeline.c#L82) | Pipe 1 provides the RGB565 video background. Pipe 2 uses the [generated input macros](../Model/NUCLEO-N657X0-Q/stai_network.h#L326) for its 256 × 256 RGB888 tensor. Width, height, channel order, and pitch establish the buffer layout. |
| 4. Get one inference image | [Snapshot and pitch branch](../Application/NUCLEO-N657X0-Q/Src/main.c#L207) → [`CameraPipeline_NNPipe_Start()`](../Application/NUCLEO-N657X0-Q/Src/app_camerapipeline.c#L148) → [frame callback](../Application/NUCLEO-N657X0-Q/Src/app_camerapipeline.c#L175) | Pipe 2 signals completion through `cameraFrameReceived`. The completed image is available in `nn_in`. When pitch differs from width × channels, the app copies the valid row bytes out of the padded buffer. |
| 5. Run the network | [Timed call site](../Application/NUCLEO-N657X0-Q/Src/main.c#L239) → [`Run_Inference()`](../Application/NUCLEO-N657X0-Q/Src/main.c#L302) → [generated `stai_network_run()`](../Model/NUCLEO-N657X0-Q/stai_network.c#L405) | The app calls the runtime in asynchronous mode and waits as requested until inference leaves the running states. The call to `stai_ext_network_new_inference()` resets execution state for the next image. The timer covers this wrapper, before postprocessing. |
| 6. Read the execution plan | [Generated epoch table](../Model/NUCLEO-N657X0-Q/network.c#L2265) and [runtime epoch runner](../Middlewares/stedgeai-lib/Npu/ll_aton/ll_aton_runtime.c#L609) | Epochs are units of scheduled network execution. The [compiler's epoch breakdown](../Model/st_ai_output/network_generate_report.txt#L748) identifies CPU and NPU operation placement within the generated execution plan. |
| 7. Pass the two raw outputs to C | [`app_postprocess_init()`](../Middlewares/ai-postprocessing-wrapper/app_postprocess_iseg_yolo_v8_ui.c#L31) → [`app_postprocess_run()`](../Middlewares/ai-postprocessing-wrapper/app_postprocess_iseg_yolo_v8_ui.c#L58) → [`iseg_yolov8_pp_process_int8()`](../Middlewares/stm32-vision-models-postprocessing/lib_vision_models_pp/Src/iseg_pp_yolov8.c#L279) | The adapter takes its parameters from `app_config.h`. `nn_out[0]` holds detections and `nn_out[1]` holds mask prototypes. The buffers contain the raw network outputs at decoder entry. |
| 8. Filter boxes and build masks | [Candidate and score filtering](../Middlewares/stm32-vision-models-postprocessing/lib_vision_models_pp/Src/iseg_pp_yolov8.c#L214) → [per-class NMS](../Middlewares/stm32-vision-models-postprocessing/lib_vision_models_pp/Src/iseg_pp_yolov8.c#L44) → [dequantization and masks](../Middlewares/stm32-vision-models-postprocessing/lib_vision_models_pp/Src/iseg_pp_yolov8.c#L108) | The decoder selects classes, suppresses overlapping boxes, and combines mask coefficients with the shared prototypes. Decoded count and objects are stored in `pp_output.nb_detect` and `pp_output.pOutBuff`. |
| 9. Draw and transmit | [`Display_NetworkOutput()`](../Application/NUCLEO-N657X0-Q/Src/main.c#L443) and [UVCL `SRCL_Update()`](../Middlewares/screenl/Src/scrl_usb.c#L485) | Normalized coordinates become display pixels. The app publishes a foreground layer with boxes, labels and masks. The USB display code composes layers for the host stream. |

**Ctrl+G** selects an explicit line in the editor. The VS Code CLI also accepts a file and line number, as shown by the following command executed from the repository root. [VS Code command-line options](https://code.visualstudio.com/docs/configure/command-line)

```powershell
code --goto "Application/NUCLEO-N657X0-Q/Src/main.c:161"
```

## 2. Inference configuration

| Setting | Definition | Effect |
| --- | --- | --- |
| RGB order, framing and camera orientation | [`COLOR_MODE`](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L40), [`ASPECT_RATIO_MODE`](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L32), [`CAMERA_FLIP`](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L27) | The bytes and spatial transform that reach the model. Consistency with host preprocessing is required for comparison of predictions. |
| Camera rate and display size | [`CAMERA_FPS` and `SCREEN_*`](../Application/NUCLEO-N657X0-Q/Inc/app_camerapipeline.h#L21) | The demo requests 30 camera frames/s and uses a 320 × 240 display canvas. The inference rate also depends on the per-frame processing time. |
| Input size, format, scale and zero point | [Generated `STAI_NETWORK_IN_1_*`](../Model/NUCLEO-N657X0-Q/stai_network.h#L326) and [input quantization](../Model/NUCLEO-N657X0-Q/stai_network.h#L340) | The reference network accepts `uint8` NHWC `[1, 256, 256, 3]`. Raw RGB works with scale ≈ 1/255 and zero point 0. A change to the input contract requires network regeneration. |
| Output shape, type and order | [Detection tensor](../Model/NUCLEO-N657X0-Q/stai_network.h#L373) and [mask prototype tensor](../Model/NUCLEO-N657X0-Q/stai_network.h#L398) | Defines how C interprets each buffer. Model substitution requires compatible layout, value semantics, and dimensions. |
| Decoder and class table | [`POSTPROCESS_TYPE`](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L35) and [`NB_CLASSES` / `CLASSES_TABLE`](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L43) | The current application uses the YOLOv8 segmentation decoder. Class indices must follow the order used by training. |
| Candidate count, prototype dimensions and output quantization | [`AI_YOLOV8_SEG_PP_*` and scale/zero-point macros](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L131) | These values describe the generated outputs. The definitions require synchronization with the detection and prototype quantization pairs in the generated header, as shown in Notebook 02. |
| Confidence, NMS IoU and object cap | [Tunable thresholds](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L141) | The defaults are 0.5 confidence, 0.4 NMS IoU and at most 10 results. Raising confidence discards more candidates. Lowering NMS IoU suppresses duplicates at less overlap. The cap also sizes firmware buffers. |
| NPU compiler options and memory | [Generation command](../Model/generate-n6-model_NUCLEO-N657X0-Q.sh#L5), [Neural-ART profile](../Model/user_neuralart_NUCLEO-N657X0-Q.json), [memory pool](../Model/my_mpools/stm32n6-app2_NUCLEO-N657X0-Q.mpool), [firmware linker region](../Application/NUCLEO-N657X0-Q/STM32CubeIDE/STM32N657xx.ld#L14) | Controls compilation and placement. The generation report and firmware linker map establish the compatibility of address ranges. |

### Effect of the confidence threshold

A controlled comparison can replay identical RGB input bytes with confidence thresholds of 0.5 and 0.7. The network output tensors are determined by the model and input, while the decoder applies the selected threshold to those tensors. Rebuilding the application incorporates the threshold change. Class-set changes require training and deployment of a compatible model with matching generated C and external constants.

## 3. Model and training configuration

YAML files describe model architecture, dataset organization, or training settings according to the interface that consumes them. The configuration location and deployment consequences differ for each category.

| Configuration category | Definition or record | Relevant fields and interpretation |
| --- | --- | --- |
| Network architecture | [Ultralytics YOLOv8-seg architecture YAML, v8.3.88](https://github.com/ultralytics/ultralytics/blob/v8.3.88/ultralytics/cfg/models/v8/yolov8-seg.yaml) and the model selected in [Notebook 01](../Notebooks/01_Train_and_Quantize_STM32N6.ipynb) | `backbone`, `head` and `scales` define architecture. The notebook starts from the pretrained `yolov8n-seg.pt` checkpoint. |
| Dataset organization | Notebook 01 writes `stm32n6_model_work/insects.yaml` beside the repository and passes it as `data=` | `path`, `train`, `val`, `test` and `names`. Class IDs in the label files index `names`. |
| Training procedure | Notebook 01's `trainer.train(...)` call and the installed [Ultralytics defaults, v8.3.88](https://github.com/ultralytics/ultralytics/blob/v8.3.88/ultralytics/cfg/default.yaml). A separate YAML can be passed as `cfg=`. | `epochs`, `batch`, `imgsz`, `optimizer`, `lr0`, augmentation and `seed`. Explicit keyword values override the corresponding YAML settings. |
| Recorded training configuration | `args.yaml` next to the run's `weights/` folder | The run record resides under `stm32n6_model_work/runs/`. Ultralytics can suffix the run directory name. Some runtime choices, such as an automatic optimizer, are best confirmed in the training log. |
| Export and compilation | Notebook 01's export cell and [Notebook 02](../Notebooks/02_Compile_and_Deploy_NUCLEO_N6.ipynb) | Export uses the chosen input size, calibration split and signed 8-bit tensors. ST generation targets the NUCLEO and requests unsigned input for the camera. |
| Embedded inference and decoding | Generated model files plus [`app_config.h`](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L131) | The trained graph and weights determine the raw tensors. The C decoder uses its own confidence threshold, NMS threshold and object cap. Deployment incorporates the compiled graph and the C configuration into the programmed images. |

### Optional training YAML

The following example defines a training configuration in `stm32n6_model_work/train_hyp.yaml`, beside the generated dataset YAML. The values establish an initial experiment configuration.

```yaml
epochs: 50         # Passes over the training data
imgsz: 256         # Input resolution matches the reference firmware
batch: 8           # Batch size determines training memory requirements
optimizer: AdamW   # Explicit optimizer so the requested lr0 is meaningful
lr0: 0.001         # Initial learning rate
seed: 42
workers: 0         # Avoids data-loader subprocesses in Windows Jupyter
```

The corresponding training call passes this file through `cfg=`. The subsequent `BEST_PT` assignment in Notebook 01 records the selected checkpoint. Explicit keyword arguments take precedence over matching YAML entries, so retaining `epochs=EPOCHS` would override the file's epoch value. A change to `imgsz` also requires synchronization of export dimensions and firmware tensor configuration. [Ultralytics configuration troubleshooting](https://docs.ultralytics.com/guides/yolo-common-issues)

```python
trainer.train(
    data=str(DATA_YAML),
    cfg=str(WORK / "train_hyp.yaml"),
    project=str(WORK / "runs"),
    name="insect_yolov8n_seg",
    device=DEVICE,
)
```

| Configuration change | Definition | Required procedure |
| --- | --- | --- |
| Filter more low-confidence boxes on the board | `AI_YOLOV8_SEG_PP_CONF_THRESHOLD` | Build and program the app |
| Change learning rate or augmentation | Training arguments or `train_hyp.yaml` | Train, evaluate, export, compile the model and deploy the matching app and weights |
| Add a new class or change the input resolution | Dataset or training/export size, then check the generated tensor contract | Train and evaluate for the new setup, export, compile and update the app and model data |
| Change the placement of model buffers | Neural-ART memory pool and firmware linker map | Regenerate, check both memory maps, rebuild and deploy the matching app and weights |
