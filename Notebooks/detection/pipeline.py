"""Train, export and compare bounding-box detectors with a shared dataset."""
import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolo11n": "yolo11n.pt",
    "mobilenetv3-large": str(Path(__file__).with_name("mobilenetv3-large.yaml").resolve()),
}


def prepare_coco(output, seed):
    """Resplit the public COCO128 training sample for a pipeline smoke run."""
    output.mkdir()
    archive = output / "coco128.zip"
    urllib.request.urlretrieve(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
        archive,
    )
    shutil.unpack_archive(archive, output)
    source = output / "coco128"
    images = sorted((source / "images/train2017").glob("*.jpg"))
    if len(images) != 128:
        raise ValueError(f"Expected 128 COCO images, found {len(images)}")
    random.Random(seed).shuffle(images)
    for split, items in zip(SPLITS, (images[:96], images[96:112], images[112:])):
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for image in items:
            shutil.copyfile(image, image_dir / image.name)
            label = source / "labels/train2017" / image.with_suffix(".txt").name
            target = label_dir / label.name
            if label.is_file():
                shutil.copyfile(label, target)
            else:
                target.touch()
    return output


def load_dataset(root, names):
    """Validate a pre-split YOLO dataset and record its image and label hashes."""
    from PIL import Image

    splits = {}
    seen = {}
    for split in SPLITS:
        image_dir = root / "images" / split
        images = sorted(
            path for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise ValueError(f"No images in {image_dir}")
        records = []
        used_labels = set()
        for image in images:
            label = root / "labels" / split / image.relative_to(image_dir).with_suffix(".txt")
            if label in used_labels:
                raise ValueError(f"Multiple images use {label}")
            used_labels.add(label)
            if not label.is_file():
                raise ValueError(f"Missing label {label}; use an empty file for a background image")
            with Image.open(image) as sample:
                sample.verify()
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            previous = seen.setdefault(image_hash, split)
            if previous != split:
                raise ValueError(f"Identical image in {previous} and {split}: {image}")
            label_bytes = label.read_bytes()
            boxes = []
            for line_number, line in enumerate(label_bytes.decode().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    cls, x, y, w, h = line.split()
                    cls = int(cls)
                    x, y, w, h = map(float, (x, y, w, h))
                    valid = (0 <= cls < len(names) and 0 < w <= 1 and 0 < h <= 1
                             and 0 <= x <= 1 and 0 <= y <= 1
                             and min(x - w / 2, y - h / 2) >= -1e-4
                             and max(x + w / 2, y + h / 2) <= 1.0001)
                    if not valid:
                        raise ValueError("box outside the allowed class or coordinate range")
                except ValueError as error:
                    raise ValueError(f"Invalid bounding box at {label}:{line_number}: {error}") from error
                boxes.append([cls, x, y, w, h])
            records.append({"image": str(image), "image_sha256": image_hash,
                            "label_sha256": hashlib.sha256(label_bytes).hexdigest(), "boxes": boxes})
        if not any(record["boxes"] for record in records):
            raise ValueError(f"No labeled objects in {split}")
        splits[split] = records
    return splits


def previews(splits, names, output, seed):
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    for split, records in splits.items():
        examples = rng.sample(records, min(6, len(records)))
        preview = Image.new("RGB", (960, 480), "#202020")
        for index, record in enumerate(examples):
            with Image.open(record["image"]) as image:
                frame = image.convert("RGB")
            frame.thumbnail((320, 240))
            width, height = frame.size
            draw = ImageDraw.Draw(frame)
            for cls, x, y, w, h in record["boxes"]:
                left, top = (x - w / 2) * width, (y - h / 2) * height
                draw.rectangle((left, top, (x + w / 2) * width, (y + h / 2) * height),
                               outline="lime", width=2)
                draw.text((left, max(0, top - 12)), names[cls], fill="lime")
            preview.paste(frame, ((index % 3) * 320, (index // 3) * 240))
        preview.save(output / f"preview-{split}.jpg")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tensor_interface(path, integer):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    if integer:
        float_tensors = [tensor["name"] for tensor in interpreter.get_tensor_details()
                         if tensor["dtype"].__name__.startswith("float")]
        if float_tensors:
            raise ValueError(f"{path} contains floating-point tensors: {float_tensors[:5]}")
    interpreter.allocate_tensors()
    interface = {}
    for kind, details in (("inputs", interpreter.get_input_details()),
                          ("outputs", interpreter.get_output_details())):
        tensors = []
        for tensor in details:
            dtype = tensor["dtype"].__name__
            scale, zero_point = tensor["quantization"]
            if integer and (dtype not in {"int8", "uint8"} or scale <= 0):
                raise ValueError(f"{path} has a non-integer or unscaled {kind} tensor: {tensor['name']}")
            tensors.append({"name": tensor["name"], "shape": tensor["shape"].tolist(),
                            "dtype": dtype, "scale": float(scale), "zero_point": int(zero_point)})
        interface[kind] = tensors
    return interface


def evaluate(artifact, data, size, directory):
    from ultralytics import YOLO

    # Use identical CPU, square inputs, batch size, and held-out images for both exports.
    result = YOLO(str(artifact), task="detect").val(
        data=str(data), split="test", imgsz=size, rect=False, batch=1, workers=0,
        device="cpu", project=str(directory), name=artifact.stem, plots=True,
    )
    return {
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "mAP50": float(result.box.map50),
        "mAP50-95": float(result.box.map),
        "inference_ms_per_image": float(result.speed["inference"]),
        "bytes": artifact.stat().st_size,
        "artifact": str(artifact),
    }


def export_tflite(model, data, size):
    from ultralytics.engine.exporter import Exporter

    class DetectionExporter(Exporter):
        def _add_tflite_metadata(self, file):
            # Ultralytics 8.3.88 AutoBackend reads the first ZIP entry of a TFLite file.
            # This keeps evaluation portable across platforms without native tflite-support.
            with zipfile.ZipFile(file, "a") as archive:
                archive.writestr("ultralytics_metadata.txt", repr(self.metadata))

    return Path(DetectionExporter(overrides={
        "format": "tflite", "imgsz": size, "batch": 1, "int8": True,
        "nms": False, "data": str(data), "split": "train", "device": "cpu",
    })(model=model.model)).resolve()


def train_and_compare(name, options, data, calibration_data, names, output):
    from ultralytics import YOLO

    directory = output / name
    model = YOLO(MODELS[name], task="detect")
    model.train(data=str(data), imgsz=options.size, epochs=options.epochs,
                batch=options.batch, workers=0, seed=options.seed, deterministic=True,
                device=options.device, project=str(directory), name="training", plots=True)
    best = Path(model.trainer.best).resolve()
    if not best.is_file():
        raise FileNotFoundError(best)

    exported = export_tflite(YOLO(str(best), task="detect"), calibration_data, options.size)
    artifacts = {
        "FP32": exported.with_name(f"{best.stem}_float32.tflite"),
        "INT8": exported.with_name(f"{best.stem}_full_integer_quant.tflite"),
    }
    report = {}
    for precision, artifact in artifacts.items():
        handoff = directory / f"{name}-{precision.lower()}.tflite"
        shutil.copyfile(artifact, handoff)
        interface = tensor_interface(handoff, integer=precision == "INT8")
        write_json(handoff.with_suffix(".json"), {
            "model": name, "precision": precision, "class_names": names,
            "input_domain": "RGB float pixels in [0, 1] before tensor quantization",
            "postprocessing": "Ultralytics box decoding and host NMS",
            **interface,
        })
        report[precision] = evaluate(handoff, data, options.size, directory / "evaluation")
    report["int8_minus_fp32_map50_95"] = report["INT8"]["mAP50-95"] - report["FP32"]["mAP50-95"]
    report["checkpoint"] = str(best)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Pre-split YOLO bounding-box dataset. Default: COCO128.")
    parser.add_argument("--classes", type=Path, help="JSON list of class names in label ID order.")
    parser.add_argument("--out", type=Path, required=True, help="New directory for this experiment.")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["yolov8n"])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-images", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prepare-only", action="store_true")
    options = parser.parse_args(argv)
    if bool(options.dataset) != bool(options.classes):
        parser.error("--dataset and --classes must be provided together")
    if options.epochs < 1 or options.batch < 1 or options.size < 32 or options.size % 32:
        parser.error("epochs and batch must be positive, and size must be a positive multiple of 32")
    if options.calibration_images < 1:
        parser.error("--calibration-images must be positive")
    if len(set(options.models)) != len(options.models):
        parser.error("list each model only once")

    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    import yaml

    output = options.out.resolve()
    if options.dataset:
        dataset = options.dataset.resolve()
        names = json.loads(options.classes.read_text(encoding="utf-8"))
    else:
        from ultralytics import __file__ as ultralytics_file

        config = Path(ultralytics_file).parent / "cfg/datasets/coco128.yaml"
        names = list(yaml.safe_load(config.read_text())["names"].values())
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        parser.error("classes must be a nonempty JSON list of distinct, nonempty strings")

    output.mkdir(parents=True)
    if not options.dataset:
        dataset = prepare_coco(output / "coco128-data", options.seed)
    splits = load_dataset(dataset, names)
    data = output / "dataset.yaml"
    dataset_config = {
        "path": str(dataset), **{split: f"images/{split}" for split in SPLITS}, "names": names,
    }
    data.write_text(yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8")
    write_json(output / "dataset-manifest.json", splits)
    write_json(output / "run.json", {
        "arguments": vars(options) | {"dataset": str(dataset), "out": str(output),
                                     "classes": names},
        "platform": platform.platform(), "python": sys.version,
        "split_counts": {split: len(records) for split, records in splits.items()},
    })
    (output / "environment.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    ))
    previews(splits, names, output, options.seed)
    if options.prepare_only:
        return
    calibration_files = output / "calibration-images.txt"
    training_files = [record["image"] for record in splits["train"]]
    selected = random.Random(options.seed).sample(
        training_files, min(len(training_files), options.calibration_images)
    )
    calibration_files.write_text("\n".join(sorted(selected)) + "\n", encoding="utf-8")
    calibration_data = output / "calibration.yaml"
    calibration_data.write_text(yaml.safe_dump(
        dataset_config | {"train": str(calibration_files)}, sort_keys=False
    ), encoding="utf-8")
    report = {}
    for name in options.models:
        report[name] = train_and_compare(name, options, data, calibration_data, names, output)
        write_json(output / "comparison.json", report)


if __name__ == "__main__":
    main()
