# Follow the C inference pipeline in VS Code

Open the repository root in VS Code. Open this file and press **Ctrl+Shift+V** on Windows to open Markdown Preview. Click a source link once to jump to its line. **Ctrl+K V** opens the preview to the side, so you can keep this guide visible while exploring the C code. The links use relative paths and `#L<number>` fragments. Each row also names the function or macro so you can find it if a source edit moves it. [VS Code Markdown preview](https://code.visualstudio.com/docs/languages/markdown) · [Source-line links](https://github.com/microsoft/vscode/issues/200709)

Once in C, put the cursor on a call or macro and use **F12** for its definition, **Alt+F12** to peek, **Shift+F12** for references, and **Alt+Left** to go back on Windows. Install Microsoft's C/C++ extension for symbol navigation. If it resolves the wrong board or cannot follow a definition, configure IntelliSense with the active NUCLEO build's compiler, include paths and defines, or its `compile_commands.json` if you have one. The links in this guide also work without C/C++ IntelliSense. [VS Code code navigation](https://code.visualstudio.com/docs/editing/editingevolved) · [Configure C/C++ IntelliSense](https://code.visualstudio.com/docs/cpp/configure-intellisense)

## 1. Open these files in execution order

Follow the rows in order for one iteration. Read the [architecture walkthrough](Application-Overview.md) alongside the source for the diagrams, tensor math and memory map.

| Stage | Jump to source | What to follow |
| --- | --- | --- |
| 1. Start the app | [`main()`](../Application/NUCLEO-N657X0-Q/Src/main.c#L161) and [`Hardware_init()`](../Application/NUCLEO-N657X0-Q/Src/main.c#L258) | The CPU initializes clocks, UART, NPU RAM and external NOR. The [memory-mapped NOR call](../Application/NUCLEO-N657X0-Q/Src/main.c#L292) makes the flashed weights accessible at their generated addresses. |
| 2. Bind the model and buffers | [`NeuralNetwork_init()`](../Application/NUCLEO-N657X0-Q/Src/main.c#L315) → [generated `stai_network_init()`](../Model/NUCLEO-N657X0-Q/stai_network.c#L359) | The app initializes the runtime and network instance. Follow `nn_in`, `nn_in_len`, `nn_out[]` and `nn_out_len[]`. The generated getters supply these buffers. |
| 3. Configure capture | [`CameraPipeline_Init()`](../Application/NUCLEO-N657X0-Q/Src/app_camerapipeline.c#L118) → [Pipe 1 settings](../Application/NUCLEO-N657X0-Q/Src/app_camerapipeline.c#L35) and [Pipe 2 settings](../Application/NUCLEO-N657X0-Q/Src/app_camerapipeline.c#L82) | Pipe 1 provides the RGB565 video background. Pipe 2 uses the [generated input macros](../Model/NUCLEO-N657X0-Q/stai_network.h#L326) for its 256 × 256 RGB888 tensor. Compare width, height, channel order and pitch. |
| 4. Get one inference image | [Snapshot and pitch branch](../Application/NUCLEO-N657X0-Q/Src/main.c#L207) → [`CameraPipeline_NNPipe_Start()`](../Application/NUCLEO-N657X0-Q/Src/app_camerapipeline.c#L148) → [frame callback](../Application/NUCLEO-N657X0-Q/Src/app_camerapipeline.c#L175) | Pipe 2 signals completion through `cameraFrameReceived`. Then check the pixel data in `nn_in`. When pitch differs from width × channels, the app copies the valid row bytes out of the padded buffer. |
| 5. Run the network | [Timed call site](../Application/NUCLEO-N657X0-Q/Src/main.c#L239) → [`Run_Inference()`](../Application/NUCLEO-N657X0-Q/Src/main.c#L302) → [generated `stai_network_run()`](../Model/NUCLEO-N657X0-Q/stai_network.c#L405) | The app calls the runtime in asynchronous mode and waits as requested until inference leaves the running states. The call to `stai_ext_network_new_inference()` resets execution state for the next image. The timer covers this wrapper, before postprocessing. |
| 6. Read the execution plan | [Generated epoch table](../Model/NUCLEO-N657X0-Q/network.c#L2265) and [runtime epoch runner](../Middlewares/stedgeai-lib/Npu/ll_aton/ll_aton_runtime.c#L609) | Epochs are units of scheduled network execution. Start with the [compiler's epoch breakdown](../Model/st_ai_output/network_generate_report.txt#L748) to see where CPU or NPU work was placed before inspecting generated code. |
| 7. Pass the two raw outputs to C | [`app_postprocess_init()`](../Middlewares/ai-postprocessing-wrapper/app_postprocess_iseg_yolo_v8_ui.c#L31) → [`app_postprocess_run()`](../Middlewares/ai-postprocessing-wrapper/app_postprocess_iseg_yolo_v8_ui.c#L58) → [`iseg_yolov8_pp_process_int8()`](../Middlewares/stm32-vision-models-postprocessing/lib_vision_models_pp/Src/iseg_pp_yolov8.c#L279) | The adapter takes its parameters from `app_config.h`. `nn_out[0]` holds detections and `nn_out[1]` holds mask prototypes. Inspect the buffers before decoding. |
| 8. Filter boxes and build masks | [Candidate and score filtering](../Middlewares/stm32-vision-models-postprocessing/lib_vision_models_pp/Src/iseg_pp_yolov8.c#L214) → [per-class NMS](../Middlewares/stm32-vision-models-postprocessing/lib_vision_models_pp/Src/iseg_pp_yolov8.c#L44) → [dequantization and masks](../Middlewares/stm32-vision-models-postprocessing/lib_vision_models_pp/Src/iseg_pp_yolov8.c#L108) | Watch how the decoder picks a class, removes overlapping boxes, and combines the selected candidate's coefficients with the shared prototypes. Inspect `pp_output.nb_detect` and `pp_output.pOutBuff` after decoding. |
| 9. Draw and transmit | [`Display_NetworkOutput()`](../Application/NUCLEO-N657X0-Q/Src/main.c#L443) and [UVCL `SRCL_Update()`](../Middlewares/screenl/Src/scrl_usb.c#L485) | Normalized coordinates become display pixels. The app publishes a foreground layer with boxes, labels and masks. The USB display code composes layers for the host stream. |

If a preview opens the file without positioning the cursor, use **Ctrl+G** and the line shown in the link, or run this from the repository root with the VS Code CLI available. [VS Code command-line options](https://code.visualstudio.com/docs/configure/command-line)

```powershell
code --goto "Application/NUCLEO-N657X0-Q/Src/main.c:161"
```

## 2. Find the settings that affect inference

| Setting | Where to look | What it controls |
| --- | --- | --- |
| RGB order, framing and camera orientation | [`COLOR_MODE`](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L40), [`ASPECT_RATIO_MODE`](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L32), [`CAMERA_FLIP`](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L27) | The bytes and spatial transform that reach the model. Compare with host preprocessing when classifications or box positions disagree. |
| Camera rate and display size | [`CAMERA_FPS` and `SCREEN_*`](../Application/NUCLEO-N657X0-Q/Inc/app_camerapipeline.h#L21) | The demo requests 30 camera frames/s and uses a 320 × 240 display canvas. The inference rate also depends on the per-frame processing time. |
| Input size, format, scale and zero point | [Generated `STAI_NETWORK_IN_1_*`](../Model/NUCLEO-N657X0-Q/stai_network.h#L326) and [input quantization](../Model/NUCLEO-N657X0-Q/stai_network.h#L340) | The reference network accepts `uint8` NHWC `[1, 256, 256, 3]`. Raw RGB works with scale ≈ 1/255 and zero point 0. Regenerate the network when changing its input contract. |
| Output shape, type and order | [Detection tensor](../Model/NUCLEO-N657X0-Q/stai_network.h#L373) and [mask prototype tensor](../Model/NUCLEO-N657X0-Q/stai_network.h#L398) | Defines how C interprets each buffer. Match layout and meaning as well as sizes when substituting a model. |
| Decoder and class table | [`POSTPROCESS_TYPE`](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L35) and [`NB_CLASSES` / `CLASSES_TABLE`](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L43) | The current application uses the YOLOv8 segmentation decoder. Class indices must follow the order used by training. |
| Candidate count, prototype dimensions and output quantization | [`AI_YOLOV8_SEG_PP_*` and scale/zero-point macros](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L131) | These values describe the generated outputs. Copy the detection and prototype quantization pairs from the new generated header, as shown in Notebook 02. |
| Confidence, NMS IoU and object cap | [Tunable thresholds](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L141) | The defaults are 0.5 confidence, 0.4 NMS IoU and at most 10 results. Raising confidence discards more candidates. Lowering NMS IoU suppresses duplicates at less overlap. The cap also sizes firmware buffers. |
| NPU compiler options and memory | [Generation command](../Model/generate-n6-model_NUCLEO-N657X0-Q.sh#L5), [Neural-ART profile](../Model/user_neuralart_NUCLEO-N657X0-Q.json), [memory pool](../Model/my_mpools/stm32n6-app2_NUCLEO-N657X0-Q.mpool), [firmware linker region](../Application/NUCLEO-N657X0-Q/STM32CubeIDE/STM32N657xx.ld#L14) | Controls compilation and placement. Check the generation report and firmware linker map together before changing memory ranges. |

**Try it with the bundled model.** Keep the same image in front of the camera, change confidence from 0.5 to 0.7 in `app_config.h`, rebuild the app, and compare the raw tensor at step 7 and the decoded count at step 8. For a controlled comparison, replay the same RGB bytes. A threshold change only affects C postprocessing and uses the same model data. The generated model data stays at the same flash address. To change what classes the model recognizes, train a model with those classes and deploy the new generated C and matching weights.

## 3. Where YOLO hyperparameters live

Yes, YAML can describe the model or supply training settings. This project uses several files with different purposes.

| Question | Owner of the setting | Example |
| --- | --- | --- |
| Which layers and model size do I use? | [Ultralytics YOLOv8-seg architecture YAML, v8.3.88](https://github.com/ultralytics/ultralytics/blob/v8.3.88/ultralytics/cfg/models/v8/yolov8-seg.yaml) and the model selected in [Notebook 01](../Notebooks/01_Train_and_Quantize_STM32N6.ipynb) | `backbone`, `head` and `scales` define architecture. The notebook starts from the pretrained `yolov8n-seg.pt` checkpoint. |
| Where are the data and labels? | Notebook 01 writes `stm32n6_model_work/insects.yaml` beside the repository and passes it as `data=` | `path`, `train`, `val`, `test` and `names`. Class IDs in the label files index `names`. |
| How do I train? | Notebook 01's `trainer.train(...)` call and the installed [Ultralytics defaults, v8.3.88](https://github.com/ultralytics/ultralytics/blob/v8.3.88/ultralytics/cfg/default.yaml). A separate YAML can be passed as `cfg=`. | `epochs`, `batch`, `imgsz`, `optimizer`, `lr0`, augmentation and `seed`. Explicit keyword values override the corresponding YAML settings. |
| Which training settings did I pass for a particular run? | `args.yaml` next to the run's `weights/` folder | Inspect the run record under `stm32n6_model_work/runs/`. Ultralytics can suffix the run directory name. Some runtime choices, such as an automatic optimizer, are best confirmed in the training log. |
| How is the trained model exported and compiled? | Notebook 01's export cell and [Notebook 02](../Notebooks/02_Compile_and_Deploy_NUCLEO_N6.ipynb) | Export uses the chosen input size, calibration split and signed 8-bit tensors. ST generation targets the NUCLEO and requests unsigned input for the camera. |
| What determines the output shown on the board? | Generated model files plus [`app_config.h`](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L131) | The trained graph and weights determine the raw tensors. The C decoder uses its own confidence threshold, NMS threshold and object cap. Editing a Python YAML won't change an existing board image. |

### Optional training YAML

Save this example as `train_hyp.yaml` beside the generated `insects.yaml` in `stm32n6_model_work/`. These are starting values for a training run.

```yaml
epochs: 50         # Passes over the training data
imgsz: 256         # Keep the first model aligned with the firmware input size
batch: 8           # Reduce if your training device runs out of memory
optimizer: AdamW   # Explicit optimizer so the requested lr0 is meaningful
lr0: 0.001         # Initial learning rate
seed: 42
workers: 0         # Useful for the Windows Jupyter workflow
```

To use it, replace the training call in Notebook 01 with this version. Keep the `BEST_PT` assignment that follows the call. Leaving `epochs=EPOCHS` in the call would override `epochs` in the file. If you change `imgsz`, also update the export size and review the firmware tensor contract. [Ultralytics configuration troubleshooting](https://docs.ultralytics.com/guides/yolo-common-issues)

```python
trainer.train(
    data=str(DATA_YAML),
    cfg=str(WORK / "train_hyp.yaml"),
    project=str(WORK / "runs"),
    name="insect_yolov8n_seg",
    device=DEVICE,
)
```

| Change you want | Where to start | Run again |
| --- | --- | --- |
| Filter more low-confidence boxes on the board | `AI_YOLOV8_SEG_PP_CONF_THRESHOLD` | Build and program the app |
| Change learning rate or augmentation | Training arguments or `train_hyp.yaml` | Train, evaluate, export, compile the model and deploy the matching app and weights |
| Add a new class or change the input resolution | Dataset or training/export size, then check the generated tensor contract | Train and evaluate for the new setup, export, compile and update the app and model data |
| Change the placement of model buffers | Neural-ART memory pool and firmware linker map | Regenerate, check both memory maps, rebuild and deploy the matching app and weights |
