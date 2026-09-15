"""Memory-bounded replacement for oemer.inference.inference.

The upstream function stores every overlapping input patch and every prediction
until the end. On a 512 MB service that causes the worker to be killed. This
version preserves the same overlap-and-average algorithm while merging each
small batch immediately.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _resize_image(image: Image.Image) -> Image.Image:
    target_pixels = int(os.getenv("OEMER_TARGET_PIXELS", "1000000"))
    width, height = image.size
    pixels = max(width * height, 1)
    ratio = (target_pixels / pixels) ** 0.5
    target_width = max(512, round(width * ratio))
    target_height = max(512, round(height * ratio))
    if (target_width, target_height) == image.size:
        return image
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _positions(length: int, window: int, step: int) -> list[int]:
    if length <= window:
        return [0]
    positions = list(range(0, length, step))
    positions = [min(position, length - window) for position in positions]
    return list(dict.fromkeys(positions))


def inference(
    model_path: str,
    img_path: str,
    step_size: int = 128,
    batch_size: int = 16,
    manual_th: Any = None,
    use_tf: bool = False,
):
    if use_tf:
        raise RuntimeError("The low-memory server supports ONNX inference only.")

    # Import this only inside the short-lived worker process. Keeping the ONNX
    # runtime and both models in the API/oemer parent processes exceeds the
    # 512 MB Render free-instance limit even when patches are streamed.
    import onnxruntime as rt

    model_dir = Path(model_path)
    with (model_dir / "metadata.pkl").open("rb") as file:
        metadata = pickle.load(file)

    options = rt.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.add_session_config_entry("session.disable_prepacking", "1")
    session = rt.InferenceSession(
        str(model_dir / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    image_cv = cv2.imread(img_path)
    if image_cv is None:
        raise ValueError(f"Could not read image: {img_path}")
    image_pil = Image.fromarray(image_cv).convert("RGB")
    image = np.asarray(_resize_image(image_pil))

    window = int(metadata["input_shape"][1])
    if image.shape[0] < window or image.shape[1] < window:
        pad_y = max(0, window - image.shape[0])
        pad_x = max(0, window - image.shape[1])
        image = np.pad(image, ((0, pad_y), (0, pad_x), (0, 0)), constant_values=255)

    output_channels = int(metadata["output_shape"][-1])
    output_sum = np.zeros((*image.shape[:2], output_channels), dtype=np.float32)
    overlap_count = np.zeros(image.shape[:2], dtype=np.uint16)
    coordinates = [
        (y, x)
        for y in _positions(image.shape[0], window, step_size)
        for x in _positions(image.shape[1], window, step_size)
    ]

    effective_batch = max(1, int(os.getenv("OEMER_BATCH_SIZE", "1")))
    total = len(coordinates)
    for start in range(0, total, effective_batch):
        current = coordinates[start : start + effective_batch]
        batch = np.stack([image[y : y + window, x : x + window] for y, x in current])
        predictions = session.run(metadata["output_names"], {"input": batch})[0]
        for prediction, (y, x) in zip(predictions, current):
            output_sum[y : y + window, x : x + window] += prediction
            overlap_count[y : y + window, x : x + window] += 1
        print(f"oemer inference: {min(start + effective_batch, total)}/{total}", flush=True)

    output_sum /= np.maximum(overlap_count[..., None], 1)
    if manual_th is None:
        class_map = np.argmax(output_sum, axis=-1).astype(np.uint8)
    else:
        class_map = np.zeros((*image.shape[:2], len(manual_th)), dtype=np.uint8)
        for index, threshold in enumerate(manual_th):
            class_map[..., index] = output_sum[..., index + 1] > threshold

    # oemer only uses the class map in its end-to-end command. Returning an
    # empty probability tensor prevents another full-image array being retained.
    return class_map, np.empty((0,), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one oemer ONNX model with bounded memory.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step-size", type=int, default=192)
    args = parser.parse_args()

    class_map, _ = inference(
        args.model_dir,
        args.image,
        step_size=args.step_size,
        batch_size=1,
    )
    np.save(args.output, class_map, allow_pickle=False)


if __name__ == "__main__":
    main()
