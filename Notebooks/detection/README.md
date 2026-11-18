# Find insects with boxes

Give the model an image. It draws a box around each insect it finds and tells you which class it belongs to. In [Notebook 03](../03_Automated_Object_Detection.ipynb), you can check the boxes, train models, and compare their results.

## Try a short run

Use Python 3.10 to 3.12. Open a terminal in the main project folder. The first run downloads sample data and model weights.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r Notebooks/detection/requirements.txt
python Notebooks/detection/pipeline.py --out detection-runs/coco-smoke
```

The model goes through the training images once. This tells you whether the whole process works. To measure how well it finds your insects, use test images that match your setup.

## Use your own images

Give each image a matching text file. For example, `01.jpg` goes with `01.txt`. Each line holds one box as `class_id x_center y_center width height`. The four numbers describe the box center and size as fractions of the image. Leave the file empty when the image has no objects. List your classes in a JSON file, for example `["moth", "fly"]`. The first class has ID 0.

Put images in `images/<capture>/` and labels in `labels/<capture>/`. Use at least three captures with labeled objects. Use `images` only as the top-level folder name. Set `--group-by-event` or `group_by_event = True` in the notebook to keep all frames from one capture together. Frames from the same capture can look almost identical, so they should stay in one split.

- **Train** teaches the model from examples.
- **Val** helps you choose the best saved version during training.
- **Test** shows how well that version works on fresh images.

The target split is about 75 percent train, 12.5 percent val, and 12.5 percent test. The exact numbers depend on the size of each capture. Check `event-splits.json` to see where each capture went. If you already have `images/{train,val,test}` and `labels/{train,val,test}`, leave out `--group-by-event`.

```bash
python Notebooks/detection/pipeline.py \
  --dataset /path/to/insects --classes /path/to/classes.json --group-by-event \
  --models yolov8n yolo11n mobilenetv3-large \
  --epochs 50 --out detection-runs/insects-01
```

Use `--prepare-only` to check the boxes, image counts, and class counts before training. Add `--min-test-images 600` to require at least 600 checked test images. Use a fresh output folder for every run.

<a id="prepare-the-ip102-detection-benchmark"></a>

## Use the public insect dataset

Put `Annotations.tar`, `JPEGImages.tar`, `test.txt`, `trainval.txt`, and `classes.txt` in `detection-runs/ip102-source/`. The two lists are in `Detection/VOC2007/ImageSets/Main`. The script converts boxes to the right format, keeps the official test split, and picks validation images from the training list. It removes exact duplicates across splits and lists them in `ip102-splits.json`. [Official download](https://github.com/xpwu95/IP102)

```bash
python Notebooks/detection/prepare_ip102.py detection-runs/ip102-source detection-runs/ip102-data
python Notebooks/detection/pipeline.py \
  --dataset detection-runs/ip102-data --classes detection-runs/ip102-data/classes.json \
  --min-test-images 600 --prepare-only --out detection-runs/ip102-inspection
```

## Add another model

Different models can take different steps to train or make a prediction. An adapter gives them a common way to return boxes, classes, and confidence scores. This lets you compare them with the same test.

Pick your models with `--models`. Add a Python adapter with `--adapter name=/path/to/file.py`. In the notebook, put the model in `models` and the file path in `adapters`. This example uses 320 pixel images for both models.

```bash
python Notebooks/detection/pipeline.py \
  --dataset /path/to/insects --classes /path/to/classes.json --group-by-event \
  --size 320 --models yolov8n ssdlite \
  --adapter ssdlite=Notebooks/detection/adapters/torchvision_ssdlite.py \
  --epochs 20 --out detection-runs/insects-adapter
```

To write your own adapter, start with the [example](adapters/torchvision_ssdlite.py) and the [shared interface](model_adapter.py). Use `context.train` and `context.val` for training and validation. Return a `TrainingResult` from `train(context)` with the saved model, how it was initialized, and the available versions (`FP32`, `FP16`, `INT8`, `NATIVE`). The predictor gets a square RGB `uint8` image in HWC order. For each object, return `[x_min, y_min, x_max, y_max, confidence, class_id]` in pixels of the padded image. Return an array of shape `(0, 6)` for no detections. Use adapter files you trust.

## Read the results

Open `comparison.json` to see the scores and `run.json` to see the settings.

- **Precision** tells you how many of the reported hits were correct.
- **Recall** tells you how many of the labeled objects the model found.
- **Box mAP** sums up how well the model finds objects and places its boxes.

Each version gets the same test images at the same size. The CPU runs one image at a time. The current timing includes work inside the adapter, such as preparing the input and reading the output. Earlier runs timed the model step alone. The report also shows how each model started, such as with weights from earlier training or a new detection head.

FP32 uses floating-point numbers. INT8 uses smaller numbers and can make the model file smaller. Compare both on the same images to see what happens to quality. The available versions depend on the adapter. To run a model on the board, you need a supported export and firmware that can read its output. Measure board timing on the board itself.
