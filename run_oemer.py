"""Run oemer on a small Render instance without overlapping heavy runtimes."""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import subprocess
import sys
import tempfile
import urllib.request
from importlib.util import find_spec
from pathlib import Path


CHECKPOINT_URLS = {
    "unet_big/model.onnx": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/1st_model.onnx",
    "seg_net/model.onnx": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/2nd_model.onnx",
}


def _oemer_module_path() -> Path:
    spec = find_spec("oemer")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("oemer is not installed.")
    return Path(next(iter(spec.submodule_search_locations)))


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    print(f"Downloading {destination.name}...", flush=True)
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    os.replace(temporary, destination)


def _ensure_onnx_models(module_path: Path) -> None:
    for relative_path, url in CHECKPOINT_URLS.items():
        destination = module_path / "checkpoints" / relative_path
        if not destination.exists() or destination.stat().st_size == 0:
            _download(url, destination)


def _run_model(model_dir: Path, image_path: Path, output_path: Path) -> None:
    worker = Path(__file__).with_name("low_memory_oemer.py")
    subprocess.run(
        [
            sys.executable,
            str(worker),
            "--model-dir",
            str(model_dir),
            "--image",
            str(image_path),
            "--output",
            str(output_path),
            "--step-size",
            os.getenv("OEMER_STEP_SIZE", "192"),
        ],
        check=True,
    )


def _prepare_prediction_cache(image_path: Path, module_path: Path) -> Path:
    """Finish both ONNX workers before importing oemer's full pipeline."""
    with tempfile.TemporaryDirectory(prefix="oemer-pred-") as workdir:
        work = Path(workdir)
        first_path = work / "first.npy"
        second_path = work / "second.npy"
        _run_model(module_path / "checkpoints/unet_big", image_path, first_path)
        _run_model(module_path / "checkpoints/seg_net", image_path, second_path)

        # Import NumPy only after the ONNX workers have exited. This keeps the
        # coordinator small while each model owns the instance memory.
        import numpy as np

        first = np.load(first_path, allow_pickle=False)
        second = np.load(second_path, allow_pickle=False)
        prediction = {
            "staff": (first == 1).astype(np.uint8),
            "symbols": (first == 2).astype(np.uint8),
            "stems_rests": (second == 1).astype(np.uint8),
            "note": (second == 2).astype(np.uint8),
            "clefs_keys": (second == 3).astype(np.uint8),
        }
        del first, second

        cache_path = image_path.with_suffix(".pkl")
        with cache_path.open("wb") as output:
            pickle.dump(prediction, output, protocol=pickle.HIGHEST_PROTOCOL)
        del prediction
        gc.collect()
        return cache_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Low-memory oemer runner")
    parser.add_argument("img_path")
    parser.add_argument("-o", "--output-path", default="./")
    parser.add_argument("-d", "--without-deskew", action="store_true")
    parser.add_argument("--use-tf", action="store_true")
    parser.add_argument("--save-cache", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    image_path = Path(args.img_path)
    if not image_path.exists():
        raise FileNotFoundError(f"The given image path does not exist: {image_path}")
    if args.use_tf:
        raise RuntimeError("The trial server supports ONNX inference only.")

    module_path = _oemer_module_path()
    _ensure_onnx_models(module_path)
    cache_path = _prepare_prediction_cache(image_path, module_path)

    # The ONNX processes are gone before the heavier oemer/scipy/sklearn stack
    # is imported, preventing their native-memory peaks from overlapping.
    import oemer.ete as ete

    ete.clear_data()
    try:
        ete.extract(args)
    finally:
        ete.clear_data()
        cache_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
