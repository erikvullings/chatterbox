import os
# Configure MPS memory allocator variables before importing torch
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.0"

import re
import shutil
import subprocess
import sys
import tempfile
import time
import torch
import soundfile as sf
import warnings
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager

# --- Suppress third-party noise during startup ---
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", message=".*resize.*does not match the required output shape.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*An output with one or more elements was resized")

# Suppress huggingface_hub progress bars on stderr (they overlap with uvicorn logs)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HUGGINGFACE_TOOLING_DISABLE_PROGRESS_BARS"] = "1"

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

# Import Chatterbox Multilingual model
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Global model cache
MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        sys.stderr.write("\033[90mLoading Chatterbox Multilingual model on device: {}...\033[0m\n".format(device))
        sys.stderr.flush()
        t0 = time.monotonic()
        MODEL = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
        elapsed = time.monotonic() - t0
        sys.stderr.write("\033[92mModel loaded in {:.1f}s\033[0m\n".format(elapsed))
        sys.stderr.flush()
    return MODEL

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup to keep it warm
    get_model()
    yield
    global MODEL
    MODEL = None

app = FastAPI(title="Chatterbox OpenAI-Compatible TTS API", lifespan=lifespan)

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str = "naive"
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0
    language_id: Optional[str] = "nl"

def split_text_into_sentences(text: str) -> list[str]:
    # Regex splits by sentence endings (. ! ?) followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # If a single sentence is still excessively long (> 200 chars), split it by commas/semicolons
        if len(s) > 200:
            sub_sentences = re.split(r'(?<=[,;])\s+', s)
            for sub in sub_sentences:
                sub = sub.strip()
                if sub:
                    result.append(sub)
        else:
            result.append(s)
    return result

def get_ffmpeg_speed_filter(speed: float) -> list[str]:
    """Generates FFmpeg atempo filters for speeds outside [0.5, 2.0] by chaining them."""
    if abs(speed - 1.0) < 0.01:
        return []
    
    filters = []
    remaining_speed = speed
    while remaining_speed > 2.0:
        filters.append("atempo=2.0")
        remaining_speed /= 2.0
    while remaining_speed < 0.5:
        filters.append("atempo=0.5")
        remaining_speed /= 0.5
    if abs(remaining_speed - 1.0) >= 0.01:
        filters.append(f"atempo={remaining_speed:.2f}")
    
    return ["-filter:a", ",".join(filters)]

@app.post("/v1/audio/speech")
async def speech_api(request: SpeechRequest):
    model = get_model()
    
    # Identify the voice path if specified or check voices/ directory
    audio_prompt_path = None
    voice_str = request.voice.strip()
    
    # 1. Check if the voice points to an existing file
    if os.path.exists(voice_str):
        audio_prompt_path = voice_str
    # 2. Check if a matching wav file exists in voices/
    elif os.path.exists(os.path.join("voices", f"{voice_str}.wav")):
        audio_prompt_path = os.path.join("voices", f"{voice_str}.wav")
    elif os.path.exists(os.path.join("voices", voice_str)):
        audio_prompt_path = os.path.join("voices", voice_str)
        
    generate_kwargs = {}
    if audio_prompt_path:
        generate_kwargs["audio_prompt_path"] = audio_prompt_path
        print(f"Using voice prompt: {audio_prompt_path}")
    else:
        print(f"Voice '{request.voice}' not found, falling back to model default voice.")

    # Split text into chunks to maintain high speech quality
    sentences = split_text_into_sentences(request.input)
    print(f"Generating speech in {len(sentences)} sentence chunks...")
    
    try:
        wav_chunks = []
        for idx, sentence in enumerate(sentences):
            wav_chunk = model.generate(
                sentence,
                language_id=request.language_id,
                **generate_kwargs
            )
            wav_chunks.append(wav_chunk)
            
        full_wav = torch.cat(wav_chunks, dim=-1)
        audio_data = full_wav.squeeze(0).cpu().numpy()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

    # Write wav data to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
        
    # Set output format details
    fmt = request.response_format.lower() if request.response_format else "mp3"
    if fmt not in ["mp3", "wav"]:
        # Fall back to mp3 if unsupported
        fmt = "mp3"
        
    # Output file
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as temp_out:
        temp_out_path = temp_out.name

    try:
        sf.write(temp_wav_path, audio_data, model.sr)
        
        # Build FFmpeg command for conversion and speed filtering
        ffmpeg_cmd = [ffmpeg_exe, "-y", "-i", temp_wav_path]
        
        # Add speed filter if any
        ffmpeg_cmd.extend(get_ffmpeg_speed_filter(request.speed))
        
        # Add codec and quality based on format
        if fmt == "mp3":
            ffmpeg_cmd.extend(["-codec:a", "libmp3lame", "-q:a", "2"])
        else: # wav
            ffmpeg_cmd.extend(["-codec:a", "pcm_s16le"])
            
        ffmpeg_cmd.append(temp_out_path)
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("FFmpeg error:")
            print(result.stderr)
            raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {result.stderr}")
            
        # Read the file to return it
        with open(temp_out_path, "rb") as out_file:
            response_content = out_file.read()
            
        media_type = "audio/mpeg" if fmt == "mp3" else "audio/wav"
        
        # Create generator to stream the response
        def iterfile():
            yield response_content
            
        return StreamingResponse(iterfile(), media_type=media_type)
        
    finally:
        # Cleanup temporary files
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        if os.path.exists(temp_out_path):
            os.remove(temp_out_path)

if __name__ == "__main__":
    import logging
    import uvicorn
    
    # Suppress uvicorn's noisy startup / connection logs
    log = logging.getLogger("uvicorn")
    log.setLevel(logging.WARNING)
    log_access = logging.getLogger("uvicorn.access")
    log_access.disabled = True
    
    sys.stderr.write("\n{}Chatterbox TTS API\033[0m\n".format("\033[92m"))
    sys.stderr.write("  {} {}\n".format("\033[90mDevice:\033[0m", device))
    sys.stderr.write("  {} {}\n".format("\033[90membed path:\033[0m", ffmpeg_exe))
    sys.stderr.write("  {}\n\n{}\033[0m".format(
        "\033[90mServer: http://127.0.0.1:8997\033[0m",
        "\033[90mLoading model in background...\033[0m"
    ))
    sys.stderr.flush()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8997,
        log_level="critical"
    )
