"""Run oemer with the memory-bounded inference implementation."""

from __future__ import annotations

import os
from pathlib import Path

import oemer.ete as ete

from low_memory_oemer import inference


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


def main() -> None:
    parser = ete.get_parser()
    args = parser.parse_args()
    if not os.path.exists(args.img_path):
        raise FileNotFoundError(f"The given image path does not exist: {args.img_path}")

    _ensure_onnx_models()
    ete.inference = inference
    ete.clear_data()
    try:
        ete.extract(args)
    finally:
        ete.clear_data()


if __name__ == "__main__":
    main()
