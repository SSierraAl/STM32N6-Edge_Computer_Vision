# Automated object detection

Open [Notebook 03](../03_Automated_Object_Detection.ipynb) to inspect image and bounding-box previews before you train. Use the same pipeline from the command line for repeatable runs. YOLOv8n and YOLO11n use pretrained detection checkpoints. MobileNetV3-Large uses ImageNet features with a newly initialized, three-scale Ultralytics detection head. Record that difference when interpreting comparisons.

## Set up and run

Use Python 3.10–3.12 with the pinned export dependencies. Run these commands from the repository root. The first run fetches COCO128 and pretrained weights.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r Notebooks/detection/requirements.txt
python Notebooks/detection/pipeline.py --out detection-runs/coco-smoke
```

The smoke run trains YOLOv8n for one epoch on a seeded 96/16/16 split of COCO128. COCO128 comes from COCO training data, so pretrained YOLO weights may have seen the held-out sample. Use an independent insect test set to report generalization.

## Use the insect dataset

Place images under `images/{train,val,test}` and matching TXT files under `labels/{train,val,test}`. Each TXT row has `class_id x_center y_center width height` with coordinates normalized by the original image dimensions. Use an empty TXT file for an image with no boxes. Supply class names as a JSON array in class-ID order, for example `["moth", "fly"]`. Keep all frames from one capture event in a single split.

```bash
python Notebooks/detection/pipeline.py \
  --dataset /path/to/insects --classes /path/to/classes.json \
  --models yolov8n yolo11n mobilenetv3-large \
  --epochs 50 --out detection-runs/insects-01
```

Use `--prepare-only` with a separate output directory to check labels, split counts and previews before training. Each invocation requires a fresh output directory. The manifest records image and label hashes. By default, up to 300 training images are selected with the run seed and recorded in `calibration-images.txt`. Adjust the limit with `--calibration-images` to represent your capture conditions. The run also records the Python environment, training arguments and training and evaluation plots.

Find the per-model scores in `comparison.json`. Both FP32 and INT8 TFLite use batch one on the same CPU and test split. Latency is Ultralytics' inference-stage mean in milliseconds per image. Use the per-tensor INT8 TFLite and its JSON tensor description for the board handoff. The repository demo decodes segmentation outputs. Integrating these bounding-box detectors on STM32N6 requires matching postprocessing and measuring latency on the board.

[Ultralytics export settings](https://docs.ultralytics.com/modes/export)
