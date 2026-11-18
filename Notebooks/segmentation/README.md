# Instance segmentation benchmark

Sebastian’s [training notebook](../01_Train_and_Quantize_STM32N6.ipynb) uses
YOLOv8n-seg. Use this pipeline to evaluate the FP32 and full INT8 TFLite exports
of the same trained checkpoint on a held-out mask dataset. The primary metric
is mask mAP50–95. The report also includes mask mAP50, mask precision and recall,
box mAP50–95, file size and model inference time on the host CPU.

Use the environment from [the detection requirements](../detection/requirements.txt).
Prepare matching `images/{train,val,test}/` and `labels/{train,val,test}/`
directories. Each label row contains a zero-based class ID followed by
normalized polygon coordinates, `class x1 y1 x2 y2 x3 y3 ...`, with one row per
object instance. Include an empty label for a background image. Give all images
from the same capture event to one split. Keep the ordered class names in a
JSON array. IP102 Detection provides boxes. Training and measuring masks
requires a dataset with instance polygon annotations.

Inspect the overlays and split before training.

```bash
python Notebooks/segmentation/pipeline.py \
  --dataset /path/to/mask-dataset --classes /path/to/classes.json \
  --prepare-only --out detection-runs/masks-inspection
```

Run the benchmark with a YOLO segmentation checkpoint.

```bash
python Notebooks/segmentation/pipeline.py \
  --dataset /path/to/mask-dataset --classes /path/to/classes.json \
  --checkpoint /path/to/yolov8n-seg.pt --epochs 20 \
  --out detection-runs/masks-benchmark
```

Repeat `--checkpoint` with another segmentation checkpoint to compare it under
the same seed, image size and test split. The default requires at least 600
validated test images. Use a lower `--min-test-images` for explicit smoke tests.
The pipeline checks polygon syntax and exact duplicate image content across the
dataset. The training split supplies INT8 calibration and the test split
supplies the final metrics. Results are stored in `comparison.json` and
per-model test plots.

At 256 × 256, the pipeline writes `insect_model.json` for signed INT8 exports
matching the demo’s tensor shapes. Point `MANIFEST` in
[Notebook 02](../02_Compile_and_Deploy_NUCLEO_N6.ipynb) to this file for ST Edge
AI generation and firmware checks. Tensor shape agreement is a host-side
check. Deployment and latency on the NUCLEO board require board verification.
