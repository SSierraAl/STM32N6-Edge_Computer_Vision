"""Convert the official IP102 Detection VOC archives to YOLO."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import tarfile
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

from PIL import Image


def image_ids(path):
    ids = path.read_text().split()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Expected unique image IDs in {path}")
    return ids


def indexed_files(archive, suffix):
    files = {}
    for member in archive:
        path = Path(member.name)
        if member.isfile() and path.suffix.lower() == suffix:
            if path.stem in files:
                raise ValueError(f"Duplicate archive member ID: {path.stem}")
            files[path.stem] = member
    return files


def convert(source, output, seed):
    test = image_ids(source / "test.txt")
    trainval = image_ids(source / "trainval.txt")
    trainval_ids = set(trainval)
    if trainval_ids.intersection(test):
        raise ValueError("The official test and trainval lists overlap")
    names = {}
    for line in (source / "classes.txt").read_text().splitlines():
        if line.strip():
            index, name = line.split(maxsplit=1)
            names[int(index) - 1] = name.strip()
    if set(names) != set(range(102)):
        raise ValueError("Expected the 102 official IP102 class names")

    repeated_annotations = []
    with tarfile.open(source / "Annotations.tar") as annotations:
        files = indexed_files(annotations, ".xml")
        labels = {}
        for image_id in test + trainval:
            with annotations.extractfile(files[image_id]) as file:
                data = file.read()
            try:
                xml = ElementTree.fromstring(data)
            except ElementTree.ParseError:
                # One source file repeats the same XML block. Accept only identical copies.
                roots = list(ElementTree.fromstring(b"<annotations>" + data + b"</annotations>"))
                if len(roots) != 2 or any(root.tag != "annotation" for root in roots):
                    raise
                for root in roots:
                    root.tail = None
                if ElementTree.tostring(roots[0]) != ElementTree.tostring(roots[1]):
                    raise
                xml = roots[0]
                repeated_annotations.append(image_id)
            width = int(xml.findtext("size/width"))
            height = int(xml.findtext("size/height"))
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions for {image_id}")
            rows = []
            for obj in xml.findall("object"):
                cls = int(obj.findtext("name"))
                if cls not in names:
                    raise ValueError(f"Unknown class {cls} in {image_id}")
                left, top, right, bottom = (
                    int(obj.findtext(f"bndbox/{field}"))
                    for field in ("xmin", "ymin", "xmax", "ymax")
                )
                # VOC counts pixels from 1 and includes both edges of the box.
                left, top = max(0, left - 1), max(0, top - 1)
                right, bottom = min(width, right), min(height, bottom)
                if right <= left or bottom <= top:
                    raise ValueError(f"Invalid box in {image_id}")
                box = ((left + right) / (2 * width), (top + bottom) / (2 * height),
                       (right - left) / width, (bottom - top) / height)
                rows.append(f"{cls} " + " ".join(f"{value:.9f}" for value in box))
            if not rows:
                raise ValueError(f"No boxes in {image_id}")
            primary_class = int(xml.findtext("object/name"))
            labels[image_id] = (width, height, primary_class, "\n".join(rows) + "\n")

    groups = defaultdict(list)
    for image_id in sorted(trainval):
        groups[labels[image_id][2]].append(image_id)
    rng = random.Random(seed)
    validation = set()
    for group in groups.values():
        rng.shuffle(group)
        validation.update(group[:min(len(group) - 1, round(len(group) * 0.1))])

    with tarfile.open(source / "JPEGImages.tar") as images:
        files = indexed_files(images, ".jpg")
        if not set(labels) <= files.keys():
            raise ValueError("The JPEG archive is missing images from the official split lists")
        output.mkdir(parents=True)
        for split in ("train", "val", "test"):
            (output / "images" / split).mkdir(parents=True)
            (output / "labels" / split).mkdir(parents=True)
        seen = {}
        counts = Counter()
        duplicate_ids = []
        # Keep the official test copy if the same image also appears in training.
        for image_id in test + [i for i in trainval if i not in validation] + sorted(validation):
            split = "test" if image_id not in trainval_ids else ("val" if image_id in validation else "train")
            with images.extractfile(files[image_id]) as image:
                data = image.read()
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen and seen[digest] != split:
                duplicate_ids.append(image_id)
                continue
            seen[digest] = split
            image_path = output / "images" / split / f"{image_id}.jpg"
            image_path.write_bytes(data)
            with Image.open(image_path) as image:
                if image.size != labels[image_id][:2]:
                    raise ValueError(f"Image and annotation dimensions differ for {image_id}")
            label_path = output / "labels" / split / f"{image_id}.txt"
            label_path.write_text(labels[image_id][3], encoding="utf-8")
            counts[split] += 1
    (output / "classes.json").write_text(json.dumps([names[i] for i in range(102)], indent=2))
    (output / "ip102-splits.json").write_text(json.dumps({
        "validation_seed": seed, "split_counts": counts,
        "cross_split_duplicates_removed": duplicate_ids,
        "identical_repeated_annotation_roots": repeated_annotations,
    }, indent=2))
    print(dict(counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ip102-", dir=args.output.parent) as directory:
        staging = Path(directory) / "dataset"
        convert(args.source, staging, args.seed)
        staging.rename(args.output)
