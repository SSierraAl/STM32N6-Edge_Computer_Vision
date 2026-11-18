"""Train, export and compare bounding-box detectors with a shared dataset."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from model_adapter import Artifact, TrainingContext, TrainingResult, ground_truth_boxes, prepare_image

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MODELS = {
    "yolov8n": ("yolov8n.pt", "COCO-pretrained YOLOv8n detection checkpoint"),
    "yolo11n": ("yolo11n.pt", "COCO-pretrained YOLO11n detection checkpoint"),
    "mobilenetv3-large": (
        str(Path(__file__).with_name("mobilenetv3-large.yaml").resolve()),
        "ImageNet-pretrained MobileNetV3-Large backbone with a newly initialized detection head",
    ),
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


def prepare_grouped(source, output, seed):
    """Copy complete capture events into a reproducible YOLO split layout."""
    events = {}
    for image in sorted((source / "images").rglob("*")):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = image.relative_to(source / "images")
        if len(relative.parts) < 2:
            raise ValueError(f"Expected images/<event>/<file>: {image}")
        if "images" in relative.parts[:-1]:
            raise ValueError(f"Rename nested folder 'images' so YOLO can find its labels: {image}")
        label = source / "labels" / relative.with_suffix(".txt")
        if not label.is_file():
            raise ValueError(f"Missing label {label}; use an empty file for a background image")
        events.setdefault(relative.parts[0], []).append((relative, label))

    order = list(events)
    random.Random(seed).shuffle(order)
    # Stable sorting uses the seed to break size ties.
    order.sort(key=lambda event: -len(events[event]))
    labeled = [event for event in order if any(
        label.read_bytes().strip() for _, label in events[event]
    )]
    if len(labeled) < len(SPLITS):
        raise ValueError("Grouped import requires at least three events with labeled objects")

    total = sum(map(len, events.values()))
    targets = dict(zip(SPLITS, (total * .75, total * .125, total * .125)))
    assignment = {labeled[0]: "train"}
    for split in ("val", "test"):
        event = min((event for event in labeled if event not in assignment),
                    key=lambda event: abs(targets[split] - len(events[event])))
        assignment[event] = split
    counts = {split: len(events[event]) for event, split in assignment.items()}
    for event in order:
        if event not in assignment:
            split = max(SPLITS, key=lambda name: targets[name] - counts[name])
            assignment[event] = split
            counts[split] += len(events[event])
        split = assignment[event]
        for relative, label in events[event]:
            image_target = output / "images" / split / relative
            label_target = output / "labels" / split / relative.with_suffix(".txt")
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / "images" / relative, image_target)
            shutil.copyfile(label, label_target)
    return assignment


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

    if integer:
        from tensorflow.lite.python import schema_py_generated as schema

        content = path.read_bytes()
        if not schema.Model.ModelBufferHasIdentifier(content, 0):
            raise ValueError(f"{path} is not a TFLite FlatBuffer")
        model = schema.Model.GetRootAsModel(content, 0)
        float_types = {value for name, value in vars(schema.TensorType).items()
                       if name.startswith(("FLOAT", "BFLOAT", "COMPLEX"))}
        for index in range(model.SubgraphsLength()):
            graph = model.Subgraphs(index)
            for tensor_index in range(graph.TensorsLength()):
                tensor = graph.Tensors(tensor_index)
                if tensor.Type() in float_types:
                    name = (tensor.Name() or b"").decode("utf-8", errors="replace")
                    raise ValueError(f"{path} contains floating-point tensor {name!r} in subgraph {index}")

    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    interface = {}
    for kind, details in (("inputs", interpreter.get_input_details()),
                          ("outputs", interpreter.get_output_details())):
        tensors = []
        for tensor in details:
            dtype = tensor["dtype"].__name__
            scale, zero_point = tensor["quantization"]
            if integer and (dtype not in {"int8", "uint8"} or not math.isfinite(scale) or scale <= 0):
                raise ValueError(f"{path} has a non-integer or unscaled {kind} tensor: {tensor['name']}")
            tensors.append({"name": tensor["name"], "shape": tensor["shape"].tolist(),
                            "dtype": dtype, "scale": float(scale), "zero_point": int(zero_point)})
        interface[kind] = tensors
    return interface


def evaluate(artifact, samples, names, size, directory):
    import numpy as np
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import DetMetrics, box_iou

    directory.mkdir(parents=True)
    validator = DetectionValidator(save_dir=directory)
    metrics = DetMetrics(save_dir=directory, plot=True, names=dict(enumerate(names)))
    matches, scores, classes, targets = [], [], [], []
    elapsed = 0.0
    for index, sample in enumerate(samples):
        frame, bounds = prepare_image(sample["image"], size)
        left, top, new_width, new_height = bounds
        if index == 0:
            artifact.predictor(frame.copy())
        start = perf_counter()
        raw = artifact.predictor(frame)
        elapsed += perf_counter() - start

        output = np.asarray(raw, dtype=np.float32)
        if output.ndim != 2 or output.shape[1] != 6 or not np.isfinite(output).all():
            raise ValueError(f"{artifact.path}: expected finite N×6 detections for {sample['image']}")
        if (np.any(output[:, 4] < 0) or np.any(output[:, 4] > 1)
                or np.any(output[:, 5] != np.floor(output[:, 5]))
                or np.any(output[:, 5] < 0) or np.any(output[:, 5] >= len(names))):
            raise ValueError(f"{artifact.path}: expected scores in [0, 1] and valid class IDs")
        output = output[output[:, 4] >= 0.001]
        output = output[np.argsort(-output[:, 4], kind="stable")[:300]]
        output[:, [0, 2]] = output[:, [0, 2]].clip(left, left + new_width)
        output[:, [1, 3]] = output[:, [1, 3]].clip(top, top + new_height)
        output = output[(output[:, 2] > output[:, 0]) & (output[:, 3] > output[:, 1])]
        boxes = np.asarray(ground_truth_boxes(sample["boxes"], bounds), dtype=np.float32).reshape(-1, 4)
        true_cls = torch.tensor([box[0] for box in sample["boxes"]], dtype=torch.int64)
        pred_cls = torch.from_numpy(output[:, 5])
        iou = box_iou(torch.from_numpy(boxes), torch.from_numpy(output[:, :4]))
        matches.append(validator.match_predictions(pred_cls, true_cls, iou).numpy())
        scores.append(output[:, 4])
        classes.append(output[:, 5])
        targets.append(true_cls.numpy())
    metrics.process(np.concatenate(matches), np.concatenate(scores),
                    np.concatenate(classes), np.concatenate(targets))
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "inference_ms_per_image": 1000 * elapsed / len(samples),
        "format": artifact.format,
        "bytes": artifact.path.stat().st_size,
        "artifact": str(artifact.path),
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


def yolo_predictor(path):
    import torch
    from ultralytics.nn.autobackend import AutoBackend
    from ultralytics.utils.ops import non_max_suppression

    backend = AutoBackend(str(path), device=torch.device("cpu"), batch=1)
    backend.eval()

    @torch.inference_mode()
    def predict(frame):
        image = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).contiguous().float() / 255
        result = backend(image)
        return non_max_suppression(result, conf_thres=0.001, iou_thres=0.7,
                                   multi_label=True, max_det=300, nc=len(backend.names),
                                   end2end=backend.end2end)[0].cpu().numpy()

    return predict


def train_ultralytics(name, context, data, calibration_data):
    from ultralytics import YOLO

    source, initialization = MODELS[name]
    model = YOLO(source, task="detect")
    model.train(data=str(data), imgsz=context.size, epochs=context.epochs,
                batch=context.batch, workers=0, seed=context.seed, deterministic=True,
                device=context.device, project=str(context.output), name="training", plots=True)
    best = Path(model.trainer.best).resolve()
    if not best.is_file():
        raise FileNotFoundError(best)

    exported = export_tflite(YOLO(str(best), task="detect"), calibration_data, context.size)
    artifacts = {}
    for precision, suffix in (("FP32", "float32"), ("INT8", "full_integer_quant")):
        artifact = exported.with_name(f"{best.stem}_{suffix}.tflite")
        handoff = context.output / f"{name}-{precision.lower()}.tflite"
        shutil.copyfile(artifact, handoff)
        artifacts[precision] = Artifact(
            handoff, yolo_predictor(handoff), "tflite",
            "RGB float pixels in [0, 1] before tensor quantization",
            "Ultralytics box decoding and host NMS",
        )
    return TrainingResult(best, initialization, artifacts)


def load_adapter(path):
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    module_name = f"detector_adapter_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    trainer = getattr(module, "train", None)
    if not callable(trainer):
        raise ValueError(f"{path}: expected train(context) returning TrainingResult")
    return trainer, digest


def train_and_compare(name, context, data, calibration_data, samples, adapter):
    if adapter:
        trainer, digest = load_adapter(adapter)
        result = trainer(context)
        source = {"file": str(adapter), "sha256": digest}
    else:
        result = train_ultralytics(name, context, data, calibration_data)
        source = {"builtin": name}
    if (not isinstance(result, TrainingResult) or not result.artifacts
            or not isinstance(result.initialization, str) or not result.initialization.strip()):
        raise ValueError(f"{name}: expected TrainingResult with artifacts and initialization")
    checkpoint = Path(result.checkpoint).resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError(f"{name}: checkpoint must be a file")
    metadata_directory = context.output
    if adapter:
        metadata_directory = context.output / "artifact-metadata"
        metadata_directory.mkdir()
    report = {}
    for precision, artifact in result.artifacts.items():
        if precision not in {"FP32", "FP16", "INT8", "NATIVE"}:
            raise ValueError(f"{name}: unsupported precision label {precision!r}")
        if not isinstance(artifact, Artifact) or not callable(artifact.predictor):
            raise ValueError(f"{name}: {precision} must be an Artifact with a predictor")
        path = artifact.path.resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"{name}: artifact must be a file")
        interface = tensor_interface(path, integer=precision == "INT8") if artifact.format == "tflite" else {}
        write_json(metadata_directory / f"{name}-{precision.lower()}.json", {
            "model": name, "precision": precision, "class_names": context.classes,
            "artifact": str(path), "format": artifact.format,
            "input_domain": artifact.preprocessing, "postprocessing": artifact.postprocessing,
            **interface,
        })
        directory = context.output / "evaluation" / f"{name}-{precision.lower()}"
        report[precision] = evaluate(artifact, samples, context.classes, context.size, directory)
    if "FP32" in report and "INT8" in report:
        report["int8_minus_fp32_map50_95"] = report["INT8"]["mAP50-95"] - report["FP32"]["mAP50-95"]
    report["checkpoint"] = str(checkpoint)
    report["initialization"] = result.initialization
    report["adapter"] = source
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="YOLO dataset, pre-split or grouped by capture event. Default: COCO128.")
    parser.add_argument("--group-by-event", action="store_true",
                        help="Split images/<event>/ and matching labels/<event>/ by event.")
    parser.add_argument("--classes", type=Path, help="JSON list of class names in label ID order.")
    parser.add_argument("--out", type=Path, required=True, help="New directory for this experiment.")
    parser.add_argument("--models", nargs="+", default=["yolov8n"],
                        help=f"Built-ins: {', '.join(MODELS)}. Additional names use --adapter.")
    parser.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH.py",
                        help="Register a trusted Python detector adapter. Repeat for multiple adapters.")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-images", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-test-images", type=int, default=1,
                        help="Require at least this many validated test images.")
    parser.add_argument("--prepare-only", action="store_true")
    options = parser.parse_args(argv)
    if options.min_test_images < 1:
        parser.error("--min-test-images must be positive")
    if bool(options.dataset) != bool(options.classes):
        parser.error("--dataset and --classes must be provided together")
    if options.group_by_event and not options.dataset:
        parser.error("--group-by-event requires --dataset and --classes")
    if min(options.epochs, options.batch, options.size) < 1:
        parser.error("epochs, batch and size must be positive")
    if options.size % 32 and any(name in MODELS for name in options.models):
        parser.error("built-in detectors require size to be a positive multiple of 32")
    if options.calibration_images < 1:
        parser.error("--calibration-images must be positive")
    if len(set(options.models)) != len(options.models):
        parser.error("list each model only once")

    adapters = {}
    for declaration in options.adapter:
        name, separator, file_name = declaration.partition("=")
        if not separator or not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            parser.error("use --adapter NAME=PATH.py with a lowercase model name")
        if name in MODELS or name in adapters or name in {"grouped-data", "coco128-data"}:
            parser.error(f"model name {name!r} is reserved or already registered")
        path = Path(file_name).expanduser().resolve()
        if path.suffix != ".py" or not path.is_file():
            parser.error(f"adapter must be an existing .py file: {path}")
        adapters[name] = path
    for name in options.models:
        if name not in MODELS and name not in adapters:
            parser.error(f"unknown model {name!r}; register it with --adapter")

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

    if options.group_by_event and output.is_relative_to(dataset):
        parser.error("--out must be outside the grouped dataset source")

    output.mkdir(parents=True)
    assignment = {}
    if options.group_by_event:
        source = dataset
        dataset = output / "grouped-data"
        assignment = prepare_grouped(source, dataset, options.seed)
    elif not options.dataset:
        dataset = prepare_coco(output / "coco128-data", options.seed)
    splits = load_dataset(dataset, names)
    test_count = len(splits["test"])
    if test_count < options.min_test_images:
        parser.error(f"Test split has {test_count} images; requires at least {options.min_test_images}")
    if assignment:
        for split, records in splits.items():
            for record in records:
                relative = Path(record["image"]).relative_to(dataset / "images" / split)
                record["event"] = relative.parts[0]
        write_json(output / "event-splits.json", assignment)

    box_counts = {}
    for split, records in splits.items():
        counts = dict.fromkeys(names, 0)
        for record in records:
            for cls, *_ in record["boxes"]:
                counts[names[cls]] += 1
        box_counts[split] = counts
    data = output / "dataset.yaml"
    dataset_config = {
        "path": str(dataset), **{split: f"images/{split}" for split in SPLITS}, "names": names,
    }
    data.write_text(yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8")
    write_json(output / "dataset-manifest.json", splits)
    write_json(output / "run.json", {
        "arguments": vars(options) | {"source": str(options.dataset.resolve()) if options.dataset else None,
                                     "dataset": str(dataset), "out": str(output),
                                     "classes": names},
        "platform": platform.platform(), "python": sys.version,
        "evaluation": {
            "protocol": "shared-bounding-box-v1", "split": "test", "batch": 1,
            "device": "cpu", "size": options.size, "min_score": 0.001,
            "max_detections": 300, "input": "RGB uint8 with aspect-preserving padding",
            "latency": "predictor call including adapter preprocessing and postprocessing",
        },
        "class_box_counts": box_counts,
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
        directory = output / name
        directory.mkdir()
        context = TrainingContext(
            directory, deepcopy(splits["train"]), deepcopy(splits["val"]), tuple(names),
            tuple(Path(path) for path in sorted(selected)), options.size,
            options.epochs, options.batch, options.seed, options.device,
        )
        report[name] = train_and_compare(
            name, context, data, calibration_data, splits["test"], adapters.get(name)
        )
        write_json(output / "comparison.json", report)


if __name__ == "__main__":
    main()
