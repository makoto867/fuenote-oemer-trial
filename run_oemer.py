"""Run oemer with the memory-bounded inference implementation."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import oemer.ete as ete


def _ensure_onnx_models() -> None:
    model_urls = {
        "unet_big/model.onnx": ete.CHECKPOINTS_URL["1st_model.onnx"],
        "seg_net/model.onnx": ete.CHECKPOINTS_URL["2nd_model.onnx"],
    }
    for relative_path, url in model_urls.items():
        destination = Path(ete.MODULE_PATH) / "checkpoints" / relative_path
        if destination.exists() and destination.stat().st_size > 0:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".download")
        ete.download_file(destination.name, url, str(temporary))
        os.replace(temporary, destination)


def _isolated_inference(
    model_path: str,
    img_path: str,
    step_size: int = 128,
    batch_size: int = 16,
    manual_th: Any = None,
    use_tf: bool = False,
):
    """Run one model in its own process so native ONNX memory is reclaimed."""
    if use_tf:
        raise RuntimeError("The trial server supports ONNX inference only.")
    if manual_th is not None:
        raise RuntimeError("Manual thresholds are not used by oemer 0.1.8 end-to-end mode.")

    worker = Path(__file__).with_name("low_memory_oemer.py")
    configured_step = int(os.getenv("OEMER_STEP_SIZE", "192"))
    with tempfile.TemporaryDirectory(prefix="oemer-model-") as workdir:
        output_path = Path(workdir) / "class_map.npy"
        subprocess.run(
            [
                sys.executable,
                str(worker),
                "--model-dir",
                model_path,
                "--image",
                img_path,
                "--output",
                str(output_path),
                "--step-size",
                str(configured_step),
            ],
            check=True,
        )
        class_map = np.load(output_path, allow_pickle=False)
    return class_map, np.empty((0,), dtype=np.float32)


def _generate_pred(img_path: str, use_tf: bool = False):
    """oemer's predictor with byte-sized masks instead of int64 masks."""
    first, _ = _isolated_inference(
        os.path.join(ete.MODULE_PATH, "checkpoints/unet_big"),
        img_path,
        use_tf=use_tf,
    )
    staff = (first == 1).astype(np.uint8)
    symbols = (first == 2).astype(np.uint8)
    del first
    gc.collect()

    second, _ = _isolated_inference(
        os.path.join(ete.MODULE_PATH, "checkpoints/seg_net"),
        img_path,
        use_tf=use_tf,
    )
    stems_rests = (second == 1).astype(np.uint8)
    notehead = (second == 2).astype(np.uint8)
    clefs_keys = (second == 3).astype(np.uint8)
    del second
    gc.collect()
    return staff, symbols, stems_rests, notehead, clefs_keys


def main() -> None:
    parser = ete.get_parser()
    args = parser.parse_args()
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"The given image path does not exist: {args.img_path}")

    _ensure_onnx_models()
    ete.inference = _isolated_inference
    ete.generate_pred = _generate_pred
    ete.clear_data()
    try:
        ete.extract(args)
    finally:
        ete.clear_data()


if __name__ == "__main__":
    main()
