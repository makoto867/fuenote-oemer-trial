from __future__ import annotations

import asyncio
import io
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError


APP_VERSION = "0.2.0"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "24000000"))
OEMER_TIMEOUT_SECONDS = int(os.getenv("OEMER_TIMEOUT_SECONDS", "1200"))

app = FastAPI(
    title="FueNote oemer trial server",
    version=APP_VERSION,
    description="楽譜画像をoemerでMusicXMLと音名列へ変換する試験API",
)
_recognition_lock = asyncio.Lock()


def _text(element: ET.Element, name: str) -> str | None:
    child = element.find(name)
    return child.text if child is not None else None


def _pitch_name(step: str, alter: int, octave: str) -> str:
    accidental = "#" * max(alter, 0) + "b" * max(-alter, 0)
    return f"{step}{accidental}{octave}"


def _solfege(step: str, alter: int, octave: str) -> str:
    names = {"C": "ド", "D": "レ", "E": "ミ", "F": "ファ", "G": "ソ", "A": "ラ", "B": "シ"}
    accidental = "♯" * max(alter, 0) + "♭" * max(-alter, 0)
    return f"{names[step]}{accidental}{octave}"


def parse_musicxml(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes)
    notes: list[dict] = []

    for measure in root.findall(".//measure"):
        measure_number = measure.attrib.get("number")
        for note in measure.findall("note"):
            duration_text = _text(note, "duration")
            base = {
                "measure": measure_number,
                "duration": int(duration_text) if duration_text else None,
                "type": _text(note, "type"),
                "dotted": note.find("dot") is not None,
                "voice": _text(note, "voice"),
                "staff": _text(note, "staff"),
                "chord": note.find("chord") is not None,
            }
            pitch = note.find("pitch")
            if pitch is None:
                notes.append({**base, "rest": True})
                continue

            step = _text(pitch, "step")
            octave = _text(pitch, "octave")
            if step is None or octave is None:
                continue
            alter_text = _text(pitch, "alter")
            alter = int(alter_text) if alter_text else 0
            notes.append(
                {
                    **base,
                    "rest": False,
                    "step": step,
                    "alter": alter,
                    "octave": int(octave),
                    "pitch": _pitch_name(step, alter, octave),
                    "solfege": _solfege(step, alter, octave),
                }
            )

    return {
        "note_count": sum(not item["rest"] for item in notes),
        "event_count": len(notes),
        "notes": notes,
    }


async def _read_limited(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="画像は15MB以下にしてください。")
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="画像ファイルが空です。")
    return b"".join(chunks)


def _normalize_image(data: bytes, output_path: Path) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="画像の解像度が大きすぎます（最大2400万画素）。")
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.save(output_path, format="PNG", optimize=True)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=415, detail="JPEG・PNGなどの画像を送信してください。") from exc


async def _run_oemer(image_path: Path, output_path: Path, without_deskew: bool) -> tuple[bytes, bytes | None]:
    runner_path = Path(__file__).with_name("run_oemer.py")
    command = [sys.executable, str(runner_path), str(image_path), "-o", str(output_path)]
    if without_deskew:
        command.append("-d")

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="oemer実行環境を開始できませんでした。") from exc

    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout=OEMER_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise HTTPException(status_code=504, detail="oemerの解析が制限時間を超えました。") from exc

    if process.returncode != 0 or not output_path.exists():
        log_tail = output.decode("utf-8", errors="replace")[-2500:]
        raise HTTPException(status_code=422, detail={"message": "oemerの解析に失敗しました。", "log": log_tail})

    teaser_path = output_path.with_name(output_path.stem + "_teaser.png")
    return output_path.read_bytes(), teaser_path.read_bytes() if teaser_path.exists() else None


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "oemer_installed": shutil.which("oemer") is not None,
        "busy": _recognition_lock.locked(),
    }


@app.post("/v1/recognize", response_model=None)
async def recognize(
    file: UploadFile = File(..., description="楽譜画像（JPEG/PNGなど）"),
    response_format: str = Query("json", pattern="^(json|musicxml)$"),
    without_deskew: bool = Query(True, description="低メモリ試験サーバーではtrueを推奨"),
) -> Response | dict:
    data = await _read_limited(file)

    async with _recognition_lock:
        with tempfile.TemporaryDirectory(prefix="oemer-") as workdir:
            work = Path(workdir)
            image_path = work / "score.png"
            output_path = work / "result.musicxml"
            await asyncio.to_thread(_normalize_image, data, image_path)
            xml_bytes, _ = await _run_oemer(image_path, output_path, without_deskew)

    if response_format == "musicxml":
        return Response(
            content=xml_bytes,
            media_type="application/vnd.recordare.musicxml+xml",
            headers={"Content-Disposition": 'attachment; filename="result.musicxml"'},
        )

    parsed = parse_musicxml(xml_bytes)
    parsed["engine"] = "oemer-0.1.8"
    parsed["musicxml"] = xml_bytes.decode("utf-8", errors="replace")
    return parsed


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FueNote oemer試験</title>
<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;max-width:760px;margin:36px auto;padding:0 18px;color:#172033}button{padding:10px 18px}pre{white-space:pre-wrap;background:#f3f5f8;padding:14px;border-radius:10px;max-height:55vh;overflow:auto}.note{color:#5d6675}</style></head>
<body><h1>FueNote oemer試験サーバー</h1><p class="note">楽譜画像を選ぶと、oemerがMusicXMLとドレミ列を返します。初回はモデル取得のため時間がかかります。</p>
<form id="form"><input id="file" type="file" accept="image/*" required> <button>解析する</button></form><p id="state"></p><pre id="result"></pre>
<script>form.onsubmit=async(e)=>{e.preventDefault();state.textContent='解析中です。画面を閉じずにお待ちください…';result.textContent='';const fd=new FormData();fd.append('file',file.files[0]);try{const r=await fetch('/v1/recognize?without_deskew=true',{method:'POST',body:fd});const text=await r.text();let j;try{j=JSON.parse(text)}catch{throw new Error(`サーバーから正常な応答がありません（HTTP ${r.status}）。再読み込みしてもう一度お試しください。`)}if(!r.ok)throw new Error(j.detail?.message||j.detail||JSON.stringify(j));state.textContent=`完了：${j.note_count}音`;result.textContent=j.notes.filter(x=>!x.rest).map(x=>x.solfege).join(' ')}catch(err){state.textContent='失敗しました';result.textContent=err instanceof TypeError?'解析サーバーとの接続が切れました。ページを再読み込みして、もう一度お試しください。':String(err.message||err)}}</script></body></html>"""
