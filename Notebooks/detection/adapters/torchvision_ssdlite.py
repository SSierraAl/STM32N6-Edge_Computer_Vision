"""Example native TorchVision SSDLite detector for the common benchmark."""
import csv
import math
import random

import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from model_adapter import Artifact, TrainingResult, ground_truth_boxes, prepare_image


def image_tensor(rgb):
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255


def read_batch(records, device):
    images, targets = [], []
    for record in records:
        rgb, bounds = prepare_image(record["image"], 320)
        boxes = ground_truth_boxes(record["boxes"], bounds)
        classes = [box[0] + 1 for box in record["boxes"]]  # Label 0 is reserved for the background.
        images.append(image_tensor(rgb).to(device))
        targets.append({
            "boxes": torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
            "labels": torch.tensor(classes, dtype=torch.int64, device=device),
        })
    return images, targets


def loss_for_split(model, records, batch, device, optimizer=None):
    total = 0.0
    for start in range(0, len(records), batch):
        subset = records[start:start + batch]
        images, targets = read_batch(subset, device)
        # Validation measures loss without changing the weights.
        with torch.set_grad_enabled(optimizer is not None):
            loss = sum(model(images, targets).values())
        value = float(loss.detach())
        if not math.isfinite(value):
            raise ValueError("SSDLite produced a non-finite loss")
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += value * len(subset)
    return total / len(records)


def train(context):
    if context.size != 320:
        raise ValueError("TorchVision SSDLite requires --size 320 for its native resolution")
    device_name = context.device
    if device_name.isdecimal():
        device_name = f"cuda:{device_name}"
    device = torch.device(device_name)
    torch.manual_seed(context.seed)
    rng = random.Random(context.seed)
    model = ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=len(context.classes) + 1,
        score_thresh=0.001, detections_per_img=300,
    ).to(device)
    model.train()
    # Freeze batch statistics so a last batch with one image can still train.
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    checkpoint = context.output / "best.pth"
    train_records = list(context.train)
    best = math.inf
    with (context.output / "loss.csv").open("w", newline="") as file:
        log = csv.writer(file)
        log.writerow(["epoch", "train_loss", "val_loss"])
        for epoch in range(1, context.epochs + 1):
            rng.shuffle(train_records)
            train_loss = loss_for_split(model, train_records, context.batch, device, optimizer)
            val_loss = loss_for_split(model, context.val, context.batch, device)
            log.writerow([epoch, train_loss, val_loss])
            if val_loss < best:
                best = val_loss
                torch.save(model.state_dict(), checkpoint)

    model.cpu()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    @torch.inference_mode()
    def predict(frame):
        detections = model([image_tensor(frame)])[0]
        # TorchVision reserves class 0 for background. Our first object class is 0.
        return torch.column_stack((
            detections["boxes"], detections["scores"], detections["labels"] - 1,
        )).numpy()

    return TrainingResult(
        checkpoint, "Random TorchVision SSDLite backbone and SSD detection head",
        {"FP32": Artifact(checkpoint, predict, "torchvision-state-dict",
                          "Square RGB float input in [0, 1]; TorchVision SSD normalization",
                          "TorchVision SSD decoding and class-wise NMS")},
    )
