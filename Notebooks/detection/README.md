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

For grouped import, use `images/<event>/<frame>.jpg` and `labels/<event>/<frame>.txt`. Give every frame from the same capture event the same top-level event folder, including frames saved in nested folders. Use at least three events with annotated objects. Choose event and subfolder names other than `images`, which YOLO uses to locate the matching labels. Each TXT row has `class_id x_center y_center width height` with normalized coordinates. Use an empty TXT file for a background image. Supply class names as a JSON array in class-ID order, for example `["moth", "fly"]`.

```bash
python Notebooks/detection/pipeline.py \
  --dataset /path/to/insects --classes /path/to/classes.json --group-by-event \
  --models yolov8n yolo11n mobilenetv3-large \
  --epochs 50 --out detection-runs/insects-01
```

The importer targets about 75% train, 12.5% validation and 12.5% test by image count. Event sizes determine the actual ratios. The run stores the split copy under `grouped-data/` and the event assignments in `event-splits.json`. Review event IDs at the source, since grouping depends on their accuracy. The existing SHA-256 check rejects byte-identical images found across splits.

For an existing dataset with `images/{train,val,test}` and matching `labels/{train,val,test}`, omit `--group-by-event`. In Notebook 03, set `group_by_event = True` for event folders. Review the per-class box counts, including zero counts, alongside the previews. Notebook training uses the grouped copy from inspection. The split is sized by image count and does not stratify classes.

Use `--prepare-only` with a separate output directory to check labels, split counts and previews before training. Use `--min-test-images 600` to require at least 600 validated test images before a thesis benchmark. Each invocation requires a fresh output directory. The manifest records image and label hashes. By default, up to 300 training images are selected with the run seed and recorded in `calibration-images.txt`. Adjust the limit with `--calibration-images` to represent your capture conditions. The run also records the Python environment, training arguments and training and evaluation plots.

## Prepare the IP102 Detection benchmark

Download `Annotations.tar`, `JPEGImages.tar` and the two lists from `Detection/VOC2007/ImageSets/Main` in the [official IP102 release](https://github.com/xpwu95/IP102). Put those four files and the official `classes.txt` into the same folder.

```bash
python Notebooks/detection/prepare_ip102.py detection-runs/ip102-source detection-runs/ip102-data
python Notebooks/detection/pipeline.py \
  --dataset detection-runs/ip102-data --classes detection-runs/ip102-data/classes.json \
  --min-test-images 600 --prepare-only --out detection-runs/ip102-inspection
```

The converter preserves the official test list and takes a seeded, class-aware validation sample from `trainval.txt`. It removes exact image duplicates across partitions, giving priority to test, then train. Split counts and removed IDs are recorded in `ip102-splits.json`.

## Add a model family

Select built-in models with `--models`. To register a detector from another family or framework, add `--adapter name=/path/to/adapter.py` and include that name in `--models`. Python adapters execute locally with your permissions. Reserve the `evaluation/` and `artifact-metadata/` directories in the model output for the pipeline. Use adapters you trust and install their dependencies into the experiment environment. For example, this runs two distinct detection families with a common 320-pixel input:

```bash
python Notebooks/detection/pipeline.py \
  --dataset /path/to/insects --classes /path/to/classes.json --group-by-event \
  --size 320 --models yolov8n ssdlite \
  --adapter ssdlite=Notebooks/detection/adapters/torchvision_ssdlite.py \
  --epochs 20 --out detection-runs/insects-yolo-ssd
```

In Notebook 03, add the name to `models`, put its file path in the `adapters` dictionary, and set the shared `size`. The included SSDLite adapter trains a TorchVision SSD detector with randomly initialized weights and saves a native FP32 checkpoint selected by validation loss. It uses 320-pixel inputs. The built-in YOLO models start from pretrained COCO detectors, and the built-in MobileNetV3-Large starts with an ImageNet backbone and a new detection head. The report records each model's initialization.

A Python adapter exports `train(context)` and returns `TrainingResult(checkpoint, initialization, artifacts)` from [model_adapter.py](model_adapter.py). Follow the [TorchVision SSDLite adapter](adapters/torchvision_ssdlite.py) as a worked example.

- `context.train` and `context.val` are sequences of dataset manifest records. Use each record's `image` path and `boxes` rows `[class_id, x_center, y_center, width, height]`. Box coordinates are normalized to the original image. Class IDs are zero-based indexes into `context.classes`. Use `context.calibration_images` for quantization and save checkpoints under `context.output`.
- The artifact mapping uses one or more of `FP32`, `FP16`, `INT8`, or `NATIVE`. Each entry specifies its on-disk file, format, predictor, and descriptions of model-specific preprocessing and decoding. Set the format to `tflite` for TFLite tensor metadata and the full integer interface checks.
- Each predictor receives a square RGB `uint8` image in HWC order. The pipeline resizes with preserved aspect ratio and centers the result on a background of value 114. Load the model once for CPU inference, synchronize before returning, and return an `N×6` array of `[x_min, y_min, x_max, y_max, confidence, class_id]` in pixels relative to the padded input. Return shape `(0, 6)` for no detections. Model-specific normalization and decoding run inside the predictor.

## Compare quality and deployment

Find per-model scores in `comparison.json` and the protocol in `run.json`. The shared evaluator gives each variant the same test images, square RGB input, CPU and batch of one. It keeps up to 300 detections with confidence at least 0.001, then uses the same IoU matching and precision, recall, mAP50 and mAP50–95 calculations across models. Model-specific decoding and suppression can differ. Reported latency covers the predictor call, including adapter preprocessing and decoding, after a warmup call. Earlier experiments used Ultralytics validation to time only the inference stage.

Use the per-tensor INT8 TFLite and its JSON tensor description for the board handoff. Available export variants depend on the adapter. The repository demo decodes segmentation outputs. An STM32N6 deployment requires a supported export, matching firmware postprocessing, and separate NPU, memory and latency measurements on the board.

[Ultralytics export settings](https://docs.ultralytics.com/modes/export)
