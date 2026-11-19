# Camera and preprocessing configuration

The NUCLEO application configures sensor support, image orientation, aspect ratio, and color order through the camera middleware and application headers. These settings affect the image presented to the network and its correspondence with the displayed scene.

## Sensor selection

The camera configuration enables IMX335, VD66GY, VD55G1, VD1943, and OV5640 sensor support through `USE_*_SENSOR` definitions. The camera middleware identifies the connected sensor during initialization. Changes to the enabled sensor set require rebuilding the application. [Camera middleware configuration](../Application/NUCLEO-N657X0-Q/Inc/cmw_camera_conf.h)

## Orientation

The `CAMERA_FLIP` definition selects the sensor orientation using the camera middleware's mirror/flip constants.

```c
#define CAMERA_FLIP CMW_MIRRORFLIP_NONE
```

The available constants are `CMW_MIRRORFLIP_NONE`, `CMW_MIRRORFLIP_FLIP`, `CMW_MIRRORFLIP_MIRROR`, and `CMW_MIRRORFLIP_FLIP_MIRROR`. Orientation forms part of the preprocessing transform and therefore affects mapping predictions back to sensor coordinates. [Application configuration](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L27)

## Aspect ratio

The camera pipeline configures display and inference independently. The reference model uses a square input, while the display uses a 320 × 240 canvas.

| Mode | Inference image | Display image |
| --- | --- | --- |
| `ASPECT_RATIO_CROP` | Cropping preserves aspect ratio while removing peripheral content to match the model dimensions. | The displayed region follows the crop mode. |
| `ASPECT_RATIO_FIT` | Resizing fills the model dimensions and can alter object aspect ratio. | The display follows the fit mode. |
| `ASPECT_RATIO_FULLSCREEN` | The inference pipe uses FIT to produce the model input. | The display maximizes the camera image within the available dimensions. |

The reference configuration selects `ASPECT_RATIO_FULLSCREEN`. The resulting inference image can therefore have a different aspect ratio from the displayed background. Prediction coordinates require the inverse of the actual inference transform when projected onto an original capture. [Aspect-ratio selection](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L29) · [Camera pipeline implementation](../Application/NUCLEO-N657X0-Q/Src/app_camerapipeline.c#L35)

## Pixel format and dimensions

Pipe 1 supplies RGB565 display pixels. Pipe 2 supplies RGB888 pixels using dimensions from the generated `STAI_NETWORK_IN_1_*` macros. `COLOR_MODE` controls the channel swap applied by the camera pipeline. The default is RGB. A mismatch between model training and camera channel order can change predictions even when tensor dimensions match.

The `CAMERA_FPS` request is 30 frames/s. The display canvas is 320 × 240. Actual inference throughput also depends on capture completion, network execution, postprocessing, and display composition. [Color configuration](../Application/NUCLEO-N657X0-Q/Inc/app_config.h#L40) · [Frame-rate and display dimensions](../Application/NUCLEO-N657X0-Q/Inc/app_camerapipeline.h#L21)
