# Find insects with masks

A box shows roughly where an insect is. A mask shows which pixels belong to it. If the image has two insects, each gets its own mask. Start with [Notebook 01](../01_Train_and_Quantize_STM32N6.ipynb) if you want to see how the training works.

## What images do I need?

For training and testing, use images with an outline drawn around each insect. Box labels alone give you no outline to compare with a predicted mask.

Put images in `images/{train,val,test}/` and the matching text files in `labels/{train,val,test}/`. Each line has a class ID and the points of one insect's outline, like `class x1 y1 x2 y2 x3 y3 ...`. The points are fractions of the image width and height. Leave the file empty for a background image. List class names in a JSON file. The first class has ID 0.

Keep frames from the same capture in one split. The pipeline requires 600 checked test images by default. Use `--min-test-images` with a smaller number for a quick run to check that the process works.

## Check the images and train

Use the same [Python setup as detection](../detection/README.md#try-a-short-run). First, check the previews to see whether the outlines sit in the right places and whether the splits make sense.

```bash
python Notebooks/segmentation/pipeline.py \
  --dataset /path/to/mask-dataset --classes /path/to/classes.json \
  --prepare-only --out detection-runs/masks-inspection
```

Start with a saved segmentation model. The command trains it on your images and exports the best saved version as FP32 and INT8.

```bash
python Notebooks/segmentation/pipeline.py \
  --dataset /path/to/mask-dataset --classes /path/to/classes.json \
  --checkpoint /path/to/yolov8n-seg.pt --epochs 20 \
  --out detection-runs/masks-benchmark
```

## Read the results

Both exports run on the same test images. **Mask mAP** shows how well the predicted masks match the drawn outlines. **Box mAP** shows how well the rectangles match. Open `comparison.json` for these scores, precision, recall, file size, and CPU inference time. INT8 calibration uses training images. The test images are set aside for the final scores.

To try another segmentation model, change the path after `--checkpoint`. Keep the same test images, image size, and seed. Give the new run its own output folder.

## Check the model for the board

At 256 × 256 pixels, the pipeline writes `insect_model.json` when the INT8 inputs and outputs have the shapes the demo expects. Set `MANIFEST` in [Notebook 02](../02_Compile_and_Deploy_NUCLEO_N6.ipynb) to that file. The notebook checks the order of the outputs and how camera pixel values are passed to the model. You can then check the masks and measure how long each image takes on the board.
