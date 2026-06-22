import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"
# Suppress huggingface_hub progress bars before any import that touches it
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", message=".*resize.*does not match the required output shape.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*An output with one or more elements was resized")
warnings.filterwarnings("ignore", message=".*torch\\.backends\\.cuda\\.sdp_kernel.*")

import torch
import soundfile as sf
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager

# Monkey patch VoiceEncoder.to to always run on CPU (workaround for PyTorch MPS LSTM bug)
from chatterbox.models.voice_encoder.voice_encoder import VoiceEncoder

original_ve_to = VoiceEncoder.to
def patched_ve_to(self, *args, **kwargs):
    return original_ve_to(self, "cpu")
VoiceEncoder.to = patched_ve_to

# Detect device and patch torch.load for MPS loading compatibility
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)
torch.load = patched_torch_load

# Find ffmpeg on the system PATH
ffmpeg_exe = shutil.which("ffmpeg")
if not ffmpeg_exe:
    raise RuntimeError("ffmpeg not found on PATH. Please install ffmpeg first.")

# Suppress tqdm globally — it creates live progress bars in logs
import tqdm as _tqdm
class _SilentTqdm(_tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("disable", True)
        super().__init__(*args, **kwargs)
_tqdm.tqdm = _SilentTqdm

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Global model cache
MODEL: Optional[ChatterboxMultilingualTTS] = None

def _log(msg: str) -> None:
    """Write a timestamped message to stderr with ANSI color — never to stdout."""
    sys.stderr.write("\033[90m[{}] {} \033[0m\n".format(
        time.strftime("%H:%M:%S"), msg))
    sys.stderr.flush()

def get_model() -> ChatterboxMultilingualTTS:
    global MODEL
    if MODEL is None:
        _log("Loading Chatterbox Multilingual model on device: {}".format(device))
        t0 = time.monotonic()
        MODEL = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
        elapsed = time.monotonic() - t0
        _log("Model loaded in {:.1f}s".format(elapsed))
    return MODEL

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()            # keep model warm during server lifetime
    yield
    global MODEL
    MODEL = None

app = FastAPI(
    title="Chatterbox OpenAI-Compatible TTS API",
    version="0.1",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SpeechRequest(BaseModel):
    model: str = "chatterbox"
    input: str = Field(description="The text to synthesize")
    voice: str = "naive"
    response_format: Optional[str] = None  # omit → server chooses
    speed: Optional[float] = 1.0
    language_id: Optional[str] = "nl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_text(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result: list[str] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(s) > 200:
            result.extend(sub.strip() for sub in re.split(r'(?<=[,;])\s+', s) if sub.strip())
        else:
            result.append(s)
    return result

def _ffmpeg_speed_filter(speed: float) -> list[str]:
    if abs(speed - 1.0) < 0.01:
        return []
    filters = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    if abs(remaining - 1.0) >= 0.01:
        filters.append(f"atempo={remaining:.2f}")
    return ["-filter:a", ",".join(filters)]

# ---------------------------------------------------------------------------
# Endpoint — returns audio with proper response_schema so the Swagger UI can
# render a **Download** button instead of garbled text in the preview pane.
# ---------------------------------------------------------------------------

@app.post(
    "/v1/audio/speech",
    responses={200: {"content": {"audio/mpeg": {}}, "description": "Synthesized audio (MP3)"}},
)
async def speech_api(request: SpeechRequest):
    model = get_model()

    # Resolve voice → prompt path
    voice_str = request.voice.strip()
    audio_prompt_path: Optional[str] = None
    for candidate in [voice_str, os.path.join("voices", f"{voice_str}.wav"), os.path.join("voices", voice_str)]:
        if os.path.exists(candidate):
            audio_prompt_path = candidate
            break

    generate_kwargs = {}
    if audio_prompt_path:
        generate_kwargs["audio_prompt_path"] = audio_prompt_path
        _log("Using voice prompt: {}".format(audio_prompt_path))
        del audio_prompt_path  # remove so we can reuse the name below without conflict

    sentences = _split_text(request.input)
    _log("Generating speech in {} sentence chunk(s)".format(len(sentences)))

    try:
        wav_chunks = []
        for sentence in sentences:
            wav_chunk = model.generate(sentence, language_id=request.language_id, **generate_kwargs)
            wav_chunks.append(wav_chunk)
        full_wav = torch.cat(wav_chunks, dim=-1)
        audio_data = full_wav.squeeze(0).cpu().numpy()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="TTS generation failed: {}".format(str(exc)))

    # ---- convert to requested format ----
    fmt = (request.response_format or "mp3").lower()
    if fmt not in ("mp3", "wav"):
        fmt = "mp3"

    media_type = "audio/mpeg" if fmt == "mp3" else "audio/wav"
    suffix     = ".mp3" if fmt == "mp3" else ".wav"

    # Write wav to temp, convert with ffmpeg in one shot
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wav_path = tmp_wav.name
    sf.write(wav_path, audio_data, model.sr)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_out:
        out_path = tmp_out.name

    try:
        cmd = [ffmpeg_exe, "-y", "-i", wav_path]
        cmd.extend(_ffmpeg_speed_filter(request.speed))
        if fmt == "mp3":
            cmd.extend(["-codec:a", "libmp3lame", "-q:a", "2"])
        else:
            cmd.extend(["-codec:a", "pcm_s16le"])
        cmd.append(out_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="FFmpeg conversion failed: {}".format(result.stderr))

        audio_bytes = Path(out_path).read_bytes()

        return StreamingResponse(
            iter([audio_bytes]),
            media_type=media_type,
            headers={"Content-Disposition": 'attachment; filename="speech.{}"'.format(fmt)},
        )
    finally:
        for p in (wav_path, out_path):
            if os.path.exists(p):
                os.remove(p)

# ---------------------------------------------------------------------------
# Main — clean stdout/stderr separation so logs never leak through API output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Silence uvicorn access logs and connection info entirely
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    banner = (
        "\n\033[92m  Chatterbox TTS API \033[0m\n"
        "  \033[90mDevice:\033[0m       {}\n"
        "  \033[90membed path:\033[0m     {}\n"
        "  \033[90mServer:\033[0m         http://127.0.0.1:8997/v1/audio/speech\n"
        "  \033[90mSwagger UI:\033[0m      http://127.0.0.1:8997/docs\n"
        "\n\033[90mLoading model …\033[0m\n".format(device, ffmpeg_exe)
    )
    sys.stderr.write(banner)
    sys.stderr.flush()

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8997, log_level="critical")
