"""Train YOLO instance segmenters and compare their FP32 and INT8 TFLite exports."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

from PIL import Image, ImageDraw
import yaml

notebooks = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(notebooks), str(notebooks / "detection")]
from detection.pipeline import IMAGE_SUFFIXES, SPLITS, export_tflite, tensor_interface, write_json


def load_masks(root, names):
    splits = {}
    seen = {}
    for split in SPLITS:
        image_dir = root / "images" / split
        records = []
        labels = set()
        for path in sorted(image_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(image_dir)
            if "images" in relative.parts[:-1] or "\n" in str(path) or "\r" in str(path):
                raise ValueError(f"Unsupported Ultralytics image path: {path}")
            label = root / "labels" / split / relative.with_suffix(".txt")
            if label in labels:
                raise ValueError(f"Multiple images use {label}")
            labels.add(label)
            content = path.read_bytes()
            with Image.open(path) as image:
                if min(image.size) < 10 or image.format not in {"JPEG", "PNG"}:
                    raise ValueError(f"Expected JPEG/PNG with dimensions of at least 10px: {path}")
                if image.format == "JPEG" and not content.endswith(b"\xff\xd9"):
                    raise ValueError(f"Expected JPEG end marker: {path}")
                image.verify()
            digest = hashlib.sha256(content).hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate image content in {seen[digest]} and {path}")
            seen[digest] = path
            annotation = label.read_bytes()
            polygons = []
            for line in annotation.decode("utf-8").strip().splitlines():
                if not line:
                    continue
                if not line.split():
                    raise ValueError(f"{label}: whitespace-only line between polygons")
                class_id, *coordinates = line.split()
                class_id = int(class_id)
                points = [float(value) for value in coordinates]
                if (not 0 <= class_id < len(names) or len(points) < 6
                        or len(points) % 2 or any(not 0 <= value <= 1 for value in points)):
                    raise ValueError(f"{label}: expected class and at least three normalized polygon points")
                vertices = list(zip(points[::2], points[1::2]))
                area = sum(x * v - y * u for (x, y), (u, v)
                           in zip(vertices, vertices[1:] + vertices[:1]))
                if math.isclose(area, 0, abs_tol=1e-12):
                    raise ValueError(f"{label}: degenerate mask polygon")
                polygons.append({"class_id": class_id, "points": vertices})
            records.append({
                "image": str(path), "image_sha256": digest,
                "label_sha256": hashlib.sha256(annotation).hexdigest(), "polygons": polygons,
            })
        if not any(record["polygons"] for record in records):
            raise ValueError(f"Expected at least one annotated mask in {split}")
        splits[split] = records
    return splits


def preview_masks(splits, names, output):
    for split, records in splits.items():
        paths = [record for record in records if record["polygons"]][:4]
        canvas = Image.new("RGB", (640, 640), (30, 30, 30))
        for index, record in enumerate(paths):
            with Image.open(record["image"]) as source:
                image = source.convert("RGB")
            image.thumbnail((320, 320))
            draw = ImageDraw.Draw(image, "RGBA")
            for polygon in record["polygons"]:
                points = [(x * image.width, y * image.height) for x, y in polygon["points"]]
                draw.polygon(points, fill=(20, 220, 180, 110), outline=(20, 220, 180, 255))
                draw.text(points[0], names[polygon["class_id"]], fill=(255, 255, 255, 255))
            canvas.paste(image, (index % 2 * 320, index // 2 * 320))
        canvas.save(output / f"preview-{split}.jpg")


def evaluate(artifact, data, directory, size, test_count):
    from ultralytics import YOLO

    model = YOLO(str(artifact), task="segment")

    def check_test_count(validator):
        if validator.seen != test_count:
            raise ValueError(f"Evaluated {validator.seen} images, expected {test_count}")

    model.add_callback("on_val_end", check_test_count)
    result = model.val(
        data=str(data), split="test", imgsz=size, rect=False, device="cpu",
        batch=1, workers=0, plots=True, project=str(directory.parent), name=directory.name,
    )
    return {
        "mask_mAP50-95": float(result.seg.map), "mask_mAP50": float(result.seg.map50),
        "mask_precision": float(result.seg.mp), "mask_recall": float(result.seg.mr),
        "box_mAP50-95": float(result.box.map), "bytes": artifact.stat().st_size,
        "cpu_inference_ms": float(result.speed["inference"]),
    }


def board_tensor_contract(interface, size, classes):
    return (
        size == 256 and len(classes) <= 255 and len(interface["inputs"]) == 1
        and interface["inputs"][0]["dtype"] == "int8"
        and interface["inputs"][0]["shape"] == [1, 256, 256, 3]
        and len(interface["outputs"]) == 2
        and all(tensor["dtype"] == "int8" for tensor in interface["outputs"])
        and sorted(tensor["shape"] for tensor in interface["outputs"])
        == sorted([[1, 4 + len(classes) + 32, 1344], [1, 64, 64, 32]])
    )


def train_compare(checkpoint, args, dataset, calibration, names, output, test_count):
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    if model.task != "segment":
        raise ValueError(f"{checkpoint}: expected a segmentation checkpoint")
    model.train(
        data=str(dataset), imgsz=args.size, epochs=args.epochs, batch=args.batch,
        seed=args.seed, device=args.device, workers=0, deterministic=True,
        plots=True, project=str(output), name="training",
    )
    best = Path(model.trainer.best)
    exported = export_tflite(YOLO(str(best)), calibration, args.size)
    artifacts = {
        "FP32": exported.with_name(f"{best.stem}_float32.tflite"),
        "INT8": exported.with_name(f"{best.stem}_full_integer_quant.tflite"),
    }
    results = {}
    for precision, path in artifacts.items():
        interface = tensor_interface(path, integer=precision == "INT8")
        metadata = {
            "imgsz": args.size, "class_names": names, "model": str(path.resolve()),
            "precision": precision, "board_tensor_contract": board_tensor_contract(
                interface, args.size, names
            ),
            **interface,
        }
        write_json(output / f"{precision.lower()}-tensors.json", metadata)
        if precision == "INT8" and metadata["board_tensor_contract"]:
            write_json(output / "insect_model.json", {
                "model": metadata["model"], "class_names": names, "imgsz": args.size,
                "input": interface["inputs"], "outputs": interface["outputs"],
            })
        results[precision] = evaluate(path, dataset, output / f"test-{precision.lower()}", args.size, test_count)
        results[precision]["model"] = metadata["model"]
    results["int8_minus_fp32_mask_mAP50-95"] = (
        results["INT8"]["mask_mAP50-95"] - results["FP32"]["mask_mAP50-95"]
    )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True, help="Ordered JSON string array")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-test-images", type=int, default=600)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch, args.size, args.min_test_images) < 1 or args.size % 32:
        parser.error("Counts must be positive and size must be a multiple of 32")
    if not args.prepare_only and not args.checkpoint:
        parser.error("--checkpoint is required for training")
    if (any(path.stem in {"", ".", ".."} for path in args.checkpoint)
            or len({path.stem for path in args.checkpoint}) != len(args.checkpoint)):
        parser.error("Checkpoints must have distinct basenames")
    names = json.loads(args.classes.read_text(encoding="utf-8"))
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names)):
        parser.error("Class names must be a non-empty JSON list of distinct strings")
    root = args.dataset.resolve()
    splits = load_masks(root, names)
    if len(splits["test"]) < args.min_test_images:
        parser.error(f"Test split has {len(splits['test'])} images, requires {args.min_test_images}")
    output = args.out.resolve()
    output.mkdir(parents=True)
    write_json(output / "manifest.json", splits)
    write_json(output / "run.json", {
        "arguments": {key: str(val) if isinstance(val, Path) else val for key, val in vars(args).items()
                      if key != "checkpoint"},
        "checkpoints": [str(path) for path in args.checkpoint],
        "images_per_split": {split: len(records) for split, records in splits.items()},
    })
    preview_masks(splits, names, output)
    if args.prepare_only:
        return
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    (output / "environment.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    ), encoding="utf-8")
    config = {"path": str(root), "names": names}
    for split, records in splits.items():
        image_list = output / f"{split}.txt"
        image_list.write_text("".join(record["image"] + "\n" for record in records), encoding="utf-8")
        config[split] = str(image_list)
    dataset = output / "dataset.yaml"
    dataset.write_text(yaml.safe_dump(config), encoding="utf-8")
    calibration = output / "calibration.yaml"
    images = [sample["image"] for sample in splits["train"]]
    selected = random.Random(args.seed).sample(images, min(300, len(images)))
    calibration_list = output / "calibration.txt"
    calibration_list.write_text("\n".join(selected) + "\n", encoding="utf-8")
    config["train"] = str(calibration_list)
    calibration.write_text(yaml.safe_dump(config), encoding="utf-8")
    results = {}
    for checkpoint in args.checkpoint:
        results[checkpoint.stem] = train_compare(
            checkpoint, args, dataset, calibration, names, output / checkpoint.stem, len(splits["test"])
        )
        write_json(output / "comparison.json", results)


if __name__ == "__main__":
    main()
