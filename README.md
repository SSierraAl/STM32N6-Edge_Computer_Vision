# STM32N6 Edge Computer Vision

This repository runs **YOLOv8n instance segmentation on the NUCLEO-N657X0-Q**. It is based on ST's [Instance Segmentation Getting Started example](https://github.com/STMicroelectronics/STM32N6-GettingStarted-InstanceSegmentation). The included model accepts a 256 × 256 RGB image and produces bounding boxes, class scores, and instance masks for 80 COCO classes.

| Goal | Start here |
| --- | --- |
| Learn the architecture from the camera to the NPU | [Architecture walkthrough](Doc/Application-Overview.md) |
| Jump to the C functions and settings in VS Code | [Source walkthrough](Doc/Source-Walkthrough.md) |
| Run the existing model on the board | [Run the bundled demo](#1-run-the-bundled-demo) |
| Train on your own images | [Notebook 01 — Train and quantize](Notebooks/01_Train_and_Quantize_STM32N6.ipynb) |
| Compile a custom model with ST Edge AI Core | [Notebook 02 — Compile and deploy](Notebooks/02_Compile_and_Deploy_NUCLEO_N6.ipynb) or [manual commands](#3-generate-the-model-files) |
| Build or update the firmware | [Build the application](#5-build-the-application) and [program external flash](#6-program-external-flash) |

## How the firmware works

The [architecture walkthrough](Doc/Application-Overview.md) traces a frame through the source code, works through a quantized detection and a mask, and shows how the firmware linker and the Neural-ART memory pool fit together.

The camera peripheral (DCMIPP) produces two image streams. Pipe 1 supplies continuous video for the USB display. Pipe 2 takes a snapshot, resizes it to the model input dimensions, and supplies RGB pixels for inference. The application runs the model with the Neural-ART runtime. The postprocessor decodes the two output tensors, filters duplicate boxes, reconstructs masks, and passes the results to the display.

| Path | Role |
| --- | --- |
| [`Application/NUCLEO-N657X0-Q/Src/main.c`](Application/NUCLEO-N657X0-Q/Src/main.c) | Camera capture, inference, postprocessing, and display loop |
| [`app_camerapipeline.c`](Application/NUCLEO-N657X0-Q/Src/app_camerapipeline.c) | Camera pipes, image dimensions, and aspect ratio handling |
| [`app_config.h`](Application/NUCLEO-N657X0-Q/Inc/app_config.h) | RGB order, class labels, confidence and IoU thresholds, output quantization |
| [`Model/NUCLEO-N657X0-Q/`](Model/NUCLEO-N657X0-Q) | Generated C/H files for the application and the model data for external flash |
| [`Model/st_ai_output/`](Model/st_ai_output) | ST Edge AI Core output and generation report. The build uses the copy in the board folder. |
| [`Middlewares/`](Middlewares) | Neural-ART runtime, camera and display libraries, and vision postprocessing |
| [`FSBL/`](FSBL) and [`Binary/`](Binary) | First-stage bootloader and prebuilt demo image |

## 1. Run the bundled demo

1. Connect the camera to the board. Connect the on-board ST-LINK port **CN9** to the PC with USB-C.
2. Set the board to **development mode** (BOOT0/JP1 at position 1 and BOOT1/JP2 at position 2). In STM32CubeProgrammer connect over SWD and select the **MX25UM51245G_STM32N6570-NUCLEO** external loader.
3. Program the [combined UVCL HEX](Binary/NUCLEO-N657X0-Q/USB-UVC-Display/NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation-uvc.hex). This image contains the bootloader, application, and COCO model data.
4. Set **both jumpers to position 1** for flash boot and power-cycle the board. Connect the USB OTG port **CN8** to the PC and open a camera viewer, such as Windows Camera. Present a COCO object to check boxes and masks.

See ST's [NUCLEO deployment guide](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/instance_segmentation/docs/README_DEPLOYMENT_STM32N6.md) for the hardware setup. The demo checks and can program the xSPI speed configuration OTP bits during initialization. The source project also contains a build configuration for the optional SPI display.

## 2. Train and export a model

Start with [Notebook 01](Notebooks/01_Train_and_Quantize_STM32N6.ipynb). It runs the existing COCO TFLite in Python, checks YOLO polygon labels, fine-tunes YOLOv8n-seg, and compares the trained PyTorch model with its full-integer TFLite export. Start with the same 256 × 256 input size and an architecture that produces the same two output tensors. Keep captures from a single event in the same dataset split.

The notebook hands over a `*_full_integer_quant.tflite` file and a JSON record with the class order and tensor details. Use representative images from your own camera for quantization calibration. [ST's YOLOv8 segmentation export tutorial](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/instance_segmentation/docs/tuto/How_to_deploy_yolov8_instance_segmentation.md) describes the per-channel export workflow.

## 3. Generate the model files

Use **ST Edge AI Core 3.0.0** to match the generated files and runtime included here. Put `stedgeai` and `arm-none-eabi-objcopy` on PATH. The commands below use Bash (Git Bash works with Windows executables on PATH). Run them from the repository root. The original TFLite is already in `Model/`. For a custom model, set `model` to its absolute path or its path relative to `Model/`.

```bash
cd Model
board=NUCLEO-N657X0-Q
model=yolov8n_256_quant_pc_ii_seg_coco-st.tflite

stedgeai generate --model "$model" --target stm32n6 \
  --st-neural-art "default@user_neuralart_${board}.json" --input-data-type uint8 && \
cp st_ai_output/network.c st_ai_output/network_ecblobs.h \
  st_ai_output/stai_network.c st_ai_output/stai_network.h "$board/" && \
cp st_ai_output/network_atonbuf.xSPI2.raw "$board/network_data.xSPI2.bin" && \
arm-none-eabi-objcopy -I binary "$board/network_data.xSPI2.bin" \
  --change-addresses 0x70380000 -O ihex "$board/network_data.hex"
cd ..
```

| Command | Why it is needed |
| --- | --- |
| `stedgeai generate` | Compiles the TFLite graph. The JSON profile sets compiler options and references the board's memory pool. `--input-data-type uint8` asks ST for an unsigned input interface to match the camera. |
| Copy the four C/H files | The application builds against the files in `Model/NUCLEO-N657X0-Q/`. |
| Copy `network_atonbuf.xSPI2.raw` | Keeps the generated data as `network_data.xSPI2.bin`. It contains the model's weights and constants. |
| `objcopy` | Wraps the raw data in Intel HEX records with base address `0x70380000`, matching the generated network's flash references. |

The repository's [generation script](Model/generate-n6-model_NUCLEO-N657X0-Q.sh) performs the same steps with the COCO filename fixed. [Notebook 02](Notebooks/02_Compile_and_Deploy_NUCLEO_N6.ipynb) generates a custom model in a separate directory so you can inspect it before copying it into the project. If upgrading ST Edge AI Core, update the application headers and runtime under `Middlewares/stedgeai-lib/` to match the generator.

## 4. Check the generated model and configure the application

Use `stai_network.h` from the **new generation** and review `network_generate_report.txt` before building.

- **Input.** The existing firmware feeds RGB888 bytes directly to the model. The COCO TFLite declares `int8` with scale ≈ 1/255 and zero point −128. ST's generated interface declares `uint8` with the same scale and zero point 0, so the raw camera bytes represent the expected normalized RGB values. Check the new model's input shape, type, scale, and zero point.
- **Outputs.** The YOLOv8 segmentation wrapper expects two signed 8-bit buffers in order. The first contains 1,344 candidates with `4 + number_of_classes + 32` values per candidate stored attribute-first. The second contains 32 mask prototypes in a 64 × 64 grid with channels last. Verify tensor order, layout, and the meaning of the values before reusing the wrapper.
- **Application settings.** Update `NB_CLASSES`, `CLASSES_TABLE`, and both output scale/zero-point pairs in [`app_config.h`](Application/NUCLEO-N657X0-Q/Inc/app_config.h). The postprocessor reads these values from the header. Change the candidate and mask dimensions or the decoder itself if the model contract changes.
- **Memory.** Inspect the actual allocations and software fallbacks in the generation report. The bundled COCO report already uses 445 of 448 KiB in one NPU SRAM bank. Moving to a higher input resolution can change the RAM requirement even when the weights still fit in flash.

## 5. Build the application

### STM32CubeIDE on Windows

1. Import [`Application/NUCLEO-N657X0-Q/STM32CubeIDE`](Application/NUCLEO-N657X0-Q/STM32CubeIDE) as an existing project and select the **UVCL** build configuration.
2. Open **Project Properties → C/C++ Build → Settings → Build Steps → Post-build steps**. The checked-in UVCL configuration contains this command. Replace the executable path with the one on your PC and ensure the build generates the input `.bin`.

   ```text
   "D:\STM32_PROG\bin\STM32_SigningTool_CLI.exe" -s -bin "${BuildArtifactFileBaseName}.bin" -o "${BuildArtifactFileBaseName}-trusted.bin" -t fsbl -la 0x70100000 -nk -hv 2.3 -align
   ```

3. Run **Clean** and **Build**. Use the resulting `*-trusted.bin` for the application region at **`0x70100000`**. The `.elf` is for debugging. In development mode, you can also load the ELF in SRAM with the model data already programmed in external flash.

The post-build step adds the STM32 v2.3 boot header. `-nk` selects the no-key flow and `-align` aligns the payload. The command above is the existing CubeIDE recipe, with `-t fsbl` and `-la` in the header. Its output is used as the **application image**. The actual first-stage bootloader is `FSBL/ai_fsbl.hex`. The `.bin` download still needs an explicit flash destination. The header's load-address field and the programmer's write address are separate settings. [ST Signing Tool manual](https://www.st.com/resource/en/user_manual/um2543-stm32-signing-tool-software-description-stmicroelectronics.pdf)

### Makefile

From the repository root, in a shell with GNU Make, Arm GCC, and STM32 Signing Tool available:

```bash
cd Application/NUCLEO-N657X0-Q
make clean
make -j8 SCR_LIB_SCREEN_ITF=UVCL
make sign SCR_LIB_SCREEN_ITF=UVCL SIGNER='STM32_SigningTool_CLI -align'
cd ../..
```

The Makefile uses the ST example's **SSBL** header recipe for the application. The override enables alignment for current Signing Tool versions that support `-align`. If you are reproducing the older toolchain and its Signing Tool does not expose this option, use the Makefile default. It produces `Application/NUCLEO-N657X0-Q/build/Application/NUCLEO-N657X0-Q/Project_sign.hex`, with the application flash address already encoded. Use one build route's application image per deployment.

## 6. Program external flash

The STM32N6 boots from external NOR. For this project's 64 MiB NUCLEO flash, the layout is:

| Region | Inclusive range | File to program |
| --- | --- | --- |
| First-stage bootloader | `0x70000000–0x700FFFFF` | `FSBL/ai_fsbl.hex` |
| Application | `0x70100000–0x7037FFFF` | CubeIDE `*-trusted.bin` at `0x70100000` or Makefile `Project_sign.hex` |
| Network weights and constants | `0x70380000–0x73FFFFFF` | `Model/NUCLEO-N657X0-Q/network_data.hex` |

At reset the bootloader starts the application. The application maps the external flash, and the generated network references its constants at the model region address. Keep the application image below `0x70380000` and the model data within the physical flash range. The weights region has 60.5 MiB of physical capacity. The compiler report derives its limit from the memory pool, so use the physical limit when checking a larger model. `.hex` files carry their write addresses. Raw `.bin` files require the address in CubeProgrammer.

1. With the board in **development mode**, connect via **SWD** and select `MX25UM51245G_STM32N6570-NUCLEO.stldr`.
2. For a fresh installation, program the FSBL, application image, and model HEX. If you changed the model, program the new application and the matching model HEX together. If you only changed firmware code and are keeping the same generated network and data, update the application region.
3. Erase only the sectors needed by the files you are updating, and verify programming. Leave **Skip flash erase before programming** unchecked for ordinary updates. It bypasses erasure and is appropriate when the target sectors are already erased. A full-chip erase clears all three regions, so all three images must be programmed again.
4. Set both jumpers to position 1 and reboot into flash mode. Check the model name on the UART console and verify boxes and masks in the USB video.

[STM32CubeProgrammer manual](https://www.st.com/resource/en/user_manual/dm00403500.pdf)

### CLI example for Windows PowerShell

Run from the repository root. Adjust the loader path. This example uses the CubeIDE application binary.

```powershell
$loader = 'D:\STM32_PROG\bin\ExternalLoader\MX25UM51245G_STM32N6570-NUCLEO.stldr'
$app = 'Application\NUCLEO-N657X0-Q\STM32CubeIDE\UVCL\NUCLEO-N657X0-Q_GettingStarted_InstanceSegmentation-trusted.bin'

STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el "$loader" -hardRst -w 'FSBL/ai_fsbl.hex' -v
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el "$loader" -hardRst -w "$app" 0x70100000 -v
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el "$loader" -hardRst -w 'Model/NUCLEO-N657X0-Q/network_data.hex' -v
```

If you built with Make, substitute `Project_sign.hex` for the application binary and omit the explicit address for that command.

### Validate a custom model

Test with both target objects and empty scenes under the same illumination used in your application. If the PC and board disagree, start with the same RGB input bytes and compare the raw output tensors before decoding. The demo's inference timer measures the network call. Measure capture and postprocessing separately to estimate end-to-end latency.
