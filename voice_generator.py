import io
import json
import os
import re
import wave
import zlib
from typing import Optional, Tuple

import requests
try:
    from murf import Murf, MurfRegion
except ImportError:  
    Murf = MurfRegion = None


DEFAULT_MODEL = os.getenv("MURF_MODEL", "FALCON")
DEFAULT_LOCALE = os.getenv("MURF_LOCALE", "en-US")
DEFAULT_SAMPLE_RATE = int(os.getenv("MURF_SAMPLE_RATE", "24000"))

AGENT_VOICES = {
    "stock_finder_agent": os.getenv("MURF_VOICE_STOCK_FINDER", "Matthew"),
    "market_data_agent": os.getenv("MURF_VOICE_MARKET_DATA", "Julia"),
    "news_analyst_agent": os.getenv("MURF_VOICE_NEWS_ANALYST", "Ken"),
    "price_recommender_agent": os.getenv("MURF_VOICE_RECOMMENDER", "Ruby"),
    "supervisor": os.getenv("MURF_VOICE_SUPERVISOR", "Emily"),
    "default": os.getenv("MURF_VOICE_ID", "Matthew"),
}



TTS_PROVIDER = os.getenv("TTS_PROVIDER", "mms").lower()
USE_MMS = TTS_PROVIDER == "mms"
MMS_MODEL_ID = os.getenv("MMS_TTS_MODEL", "facebook/mms-tts-eng")

MMS_VOICE_PROFILES = {
    "Matthew": {"shift": -4.0, "rate": 0.95, "noise": 0.45, "noise_dur": 0.50, "tilt": -2.0, "seed": 11},   # stock finder: deep, steady
    "Julia":   {"shift": 3.5,  "rate": 1.10, "noise": 0.80, "noise_dur": 0.80, "tilt": 2.0,  "seed": 23},   # market data: bright, brisk
    "Ken":     {"shift": -7.5, "rate": 0.88, "noise": 0.40, "noise_dur": 0.40, "tilt": -4.0, "seed": 37},   # news analyst: very deep, slow
    "Ruby":    {"shift": 6.5,  "rate": 1.00, "noise": 0.95, "noise_dur": 0.95, "tilt": 3.0,  "seed": 41},   # recommender: high, expressive
    "Emily":   {"shift": 1.5,  "rate": 0.97, "noise": 0.60, "noise_dur": 0.60, "tilt": 0.0,  "seed": 59},   # supervisor: calm, mid
}
_mms_cache = {}

_DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _spell_numbers(text: str) -> str:
    """MMS-TTS can't read digits, so spell numbers out as words."""
    try:
        from num2words import num2words
    except ImportError:
        num2words = None

    def repl(match):
        raw = match.group(0).replace(",", "")
        if num2words:
            try:
                return " " + num2words(float(raw) if "." in raw else int(raw)) + " "
            except Exception:
                pass
        return " " + " ".join("point" if c == "." else _DIGIT_WORDS[int(c)] for c in raw) + " "

    return re.sub(r"\d+(?:,\d{3})*(?:\.\d+)?", repl, text)


def _clean_for_mms(text: str) -> str:
    """Strip markdown/emoji/URLs and normalise symbols so MMS-TTS can speak the text."""
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("\u20b9", " rupees ").replace("Rs.", " rupees ")
    text = text.replace("%", " percent ").replace("&", " and ")
    text = _spell_numbers(text)
    text = re.sub(r"[^A-Za-z'.,!?;:\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _mms_load():
    if "model" not in _mms_cache:
        import torch
        from transformers import AutoTokenizer, VitsModel
        _mms_cache["torch"] = torch
        _mms_cache["tokenizer"] = AutoTokenizer.from_pretrained(MMS_MODEL_ID)
        _mms_cache["model"] = VitsModel.from_pretrained(MMS_MODEL_ID).eval()
    return _mms_cache["model"], _mms_cache["tokenizer"], _mms_cache["torch"]


def _voice_profile(voice_id: str) -> dict:
    """Profile for a voice name; unknown names get a stable, hash-derived profile so they still differ."""
    profile = MMS_VOICE_PROFILES.get(voice_id)
    if profile:
        return profile
    h = zlib.crc32(str(voice_id).encode())
    return {
        "shift": (h % 15) - 7.0,
        "rate": 0.88 + ((h >> 4) % 25) / 100.0,
        "noise": 0.40 + ((h >> 8) % 56) / 100.0,
        "noise_dur": 0.40 + ((h >> 12) % 56) / 100.0,
        "tilt": ((h >> 16) % 9) - 4.0,
        "seed": h % 1000,
    }


def _apply_tilt(audio, sr: int, tilt_db_per_octave: float):
    """Tilt the spectrum around 1 kHz: negative = warmer/darker voice, positive = brighter voice."""
    import numpy as np
    if not tilt_db_per_octave:
        return audio
    spectrum = np.fft.rfft(audio)
    freqs = np.maximum(np.fft.rfftfreq(len(audio), 1.0 / sr), 100.0)
    gain_db = np.clip(tilt_db_per_octave * np.log2(freqs / 1000.0), -12.0, 12.0)
    return np.fft.irfft(spectrum * (10.0 ** (gain_db / 20.0)), n=len(audio))


def _mms_synthesize(text: str, voice_id: str) -> bytes:
    """Synthesize text with Meta MMS-TTS and return mono 16-bit WAV bytes at DEFAULT_SAMPLE_RATE."""
    import numpy as np

    model, tokenizer, torch = _mms_load()
    sr = model.config.sampling_rate

    profile = _voice_profile(voice_id)
    factor = 2 ** (profile["shift"] / 12.0)
    model.speaking_rate = profile["rate"] / factor
    model.noise_scale = profile["noise"]
    model.noise_scale_duration = profile["noise_dur"]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", _clean_for_mms(text)) if s.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > 250:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)

    gap = np.zeros(int(sr * 0.25 / profile["rate"]), dtype=np.float32)
    pieces = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt")
        if inputs["input_ids"].shape[1] == 0:
            continue
        torch.manual_seed(profile["seed"])  
        with torch.no_grad():
            wav = model(**inputs).waveform[0].cpu().numpy().astype(np.float32)
        pieces.extend([wav, gap])
    if not pieces:
        raise ValueError("No speakable text after cleaning.")

    audio = np.concatenate(pieces)
    out_len = max(1, int(len(audio) * (DEFAULT_SAMPLE_RATE / sr) / factor))
    audio = np.interp(np.linspace(0, len(audio) - 1, out_len), np.arange(len(audio)), audio)
    audio = _apply_tilt(audio, DEFAULT_SAMPLE_RATE, profile["tilt"])

    rms = float(np.sqrt(np.mean(audio ** 2))) or 1.0
    audio = audio * (0.10 / rms)
    peak = float(np.max(np.abs(audio))) or 1.0
    if peak > 0.95:
        audio = audio * (0.95 / peak)
    pcm = (audio * 32767).astype("<i2").tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(DEFAULT_SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def _tts_stream(client, text: str, voice_id: str):
    """Return an iterable of audio byte chunks from the configured TTS provider."""
    if USE_MMS:
        return [_mms_synthesize(text, voice_id)]
    return client.text_to_speech.stream(
        text=text,
        voice_id=voice_id,
        model=DEFAULT_MODEL,
        multi_native_locale=DEFAULT_LOCALE,
        sample_rate=DEFAULT_SAMPLE_RATE
    )


def format_agent_name(name: str) -> str:
    """Format agent name for display by replacing underscores and title-casing."""
    return name.replace('_', ' ').title()


def get_murf_client():
    """Get Murf client with API key from environment."""
    api_key = os.getenv("MURF_API_KEY")
    if not api_key or Murf is None:
        return None
    
    return Murf(
        api_key=api_key,
        region=MurfRegion.GLOBAL
    )


def generate_voice_output(text: str, output_file: str = "output.wav", voice_id: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate voice output from text using Murf AI.

    Args:
        text: The text to convert to speech
        output_file: Path to save the audio file
        voice_id: Optional voice ID to use (defaults to AGENT_VOICES["default"])

    Returns:
        tuple: (output_file_path, error_message) - output_file_path is the path if successful,
               error_message contains details if generation fails. One of them will be None.
    """
    client = get_murf_client()
    if not client and not USE_MMS:
        error_msg = "Voice generation requires MURF_API_KEY to be set in environment variables."
        print(f"Warning: {error_msg}")
        return (None, error_msg)

    if not text or not text.strip():
        error_msg = "No text provided for voice generation."
        print(f"Warning: {error_msg}")
        return (None, error_msg)

    voice = voice_id or AGENT_VOICES["default"]

    try:
        audio_stream = _tts_stream(client, text, voice)

        with open(output_file, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        print(f"Voice output saved to: {output_file}")
        return (output_file, None)
    except IOError as e:
        error_msg = f"Error writing audio file: {e}"
        print(error_msg)
        return (None, error_msg)
    except Exception as e:
        error_msg = f"Voice generation failed: {str(e)}"
        print(error_msg)
        return (None, error_msg)


def generate_multi_agent_debate(agent_messages: list, output_file: str = "Sensei4Stocks/stock_debate.wav") -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a multi-agent debate voice output where each agent has a unique voice.
    
    Args:
        agent_messages: List of tuples (agent_name, message_content)
                       e.g., [("stock_finder_agent", "I found these stocks..."), ...]
        output_file: Path to save the combined audio file
    
    Returns:
        tuple: (output_file_path, error_message) - output_file_path is the path if successful,
               error_message contains details if generation fails. One of them will be None.
    """
    client = get_murf_client()
    if not client and not USE_MMS:
        error_msg = "Voice generation requires MURF_API_KEY to be set in environment variables."
        print(f"Warning: {error_msg}")
        return (None, error_msg)

    if not agent_messages:
        error_msg = "No agent messages to generate voice for."
        print(error_msg)
        return (None, error_msg)

    try:
        with open(output_file, "wb") as f:
            for agent_name, content in agent_messages:
                if not content or not content.strip():
                    continue
                    
                voice_id = AGENT_VOICES.get(agent_name, AGENT_VOICES["default"])
                
                intro_text = f"{format_agent_name(agent_name)} speaking: "
                full_text = intro_text + content
                
                print(f"Generating voice for {agent_name} using voice {voice_id}...")
                
                audio_stream = _tts_stream(client, full_text, voice_id)
                
                for chunk in audio_stream:
                    f.write(chunk)
        
        print(f"Multi-agent debate voice output saved to: {output_file}")
        return (output_file, None)
    except IOError as e:
        error_msg = f"Error writing audio file: {e}"
        print(error_msg)
        return (None, error_msg)
    except Exception as e:
        error_msg = f"Voice generation failed: {str(e)}"
        print(error_msg)
        return (None, error_msg)


def generate_individual_agent_audio(agent_messages: list, output_dir: str = ".") -> Tuple[Optional[dict], Optional[str]]:
    """
    Generate individual audio files for each agent's message.
    
    Args:
        agent_messages: List of tuples (agent_name, message_content)
                       e.g., [("stock_finder_agent", "I found these stocks..."), ...]
        output_dir: Directory to save the individual audio files
    
    Returns:
        tuple: (dict of {agent_name: file_path}, error_message) - dict is the mapping if successful,
               error_message contains details if generation fails.
    """
    client = get_murf_client()
    if not client and not USE_MMS:
        error_msg = "Voice generation requires MURF_API_KEY to be set in environment variables."
        print(f"Warning: {error_msg}")
        return (None, error_msg)

    if not agent_messages:
        error_msg = "No agent messages to generate voice for."
        print(error_msg)
        return (None, error_msg)

    agent_audio_files = {}
    
    try:
        for agent_name, content in agent_messages:
            if not content or not content.strip():
                continue
                
            voice_id = AGENT_VOICES.get(agent_name, AGENT_VOICES["default"])
            
            intro_text = f"{format_agent_name(agent_name)} speaking: "
            full_text = intro_text + content
            
            output_file = os.path.join(output_dir, f"{agent_name}_audio.wav")
            
            print(f"Generating voice for {agent_name} using voice {voice_id}...")
            
            audio_stream = _tts_stream(client, full_text, voice_id)
            
            with open(output_file, "wb") as f:
                for chunk in audio_stream:
                    f.write(chunk)
            
            agent_audio_files[agent_name] = output_file
            print(f"Audio saved for {agent_name}: {output_file}")
        
        return (agent_audio_files, None)
    except IOError as e:
        error_msg = f"Error writing audio file: {e}"
        print(error_msg)
        return (None, error_msg)
    except Exception as e:
        error_msg = f"Voice generation failed: {str(e)}"
        print(error_msg)
        return (None, error_msg)


META_ASR_URL = os.getenv("META_ASR_URL", "https://api.meta.ai/v1/asr/transcribe")
META_ASR_MODEL = os.getenv("META_ASR_MODEL", "muse-voice-transcribe-1.0")
META_ASR_SAMPLE_RATE = 24000  


def _to_meta_wav(audio_data: bytes) -> bytes:
    """
    Convert WAV bytes to mono, 16-bit PCM at 24 kHz, which is what the
    Meta Model API transcribe endpoint accepts. Returns the input unchanged
    if it can't be parsed (the API will then report the problem).
    """
    try:
        import numpy as np

        with wave.open(io.BytesIO(audio_data), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if width != 2:
            return audio_data

        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

        if rate != META_ASR_SAMPLE_RATE and len(samples) > 1:
            if rate > META_ASR_SAMPLE_RATE:
                k = max(1, int(round(rate / META_ASR_SAMPLE_RATE)))
                if k > 1:
                    samples = np.convolve(samples, np.ones(k, dtype=np.float32) / k, mode="same")
            new_len = int(round(len(samples) * META_ASR_SAMPLE_RATE / rate))
            old_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
            new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
            samples = np.interp(new_x, old_x, samples)

        pcm = np.clip(samples, -32768, 32767).astype("<i2").tobytes()

        out = io.BytesIO()
        with wave.open(out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(META_ASR_SAMPLE_RATE)
            wf.writeframes(pcm)
        return out.getvalue()
    except Exception as e:
        print(f"Audio conversion skipped: {e}")
        return audio_data


def transcribe_audio(audio_data: bytes, voice_id: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribe audio to text using Meta's Muse Voice Transcribe (Meta Model API).

    Args:
        audio_data: Audio data as bytes (WAV format)
        voice_id: Unused. Kept for backward compatibility with existing callers.

    Returns:
        tuple: (transcribed_text, error_message) - transcribed_text is the text if successful,
               error_message contains details if transcription fails. One of them will be None.
    """
    api_key = os.getenv("MODEL_API_KEY")
    if not api_key:
        error_msg = "Speech recognition requires MODEL_API_KEY (Meta Model API key) to be set in environment variables."
        print(f"Warning: {error_msg}")
        return (None, error_msg)

    if not audio_data:
        return (None, "No audio data provided.")

    wav_bytes = _to_meta_wav(audio_data)

    try:
        response = requests.post(
            META_ASR_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            files={
                "request": (
                    None,
                    json.dumps({
                        "model": META_ASR_MODEL,
                        "audioEncoding": "WAV",
                        "mode": "PUSH_TO_TALK",
                    }),
                    "application/json",
                ),
                "audio": ("recording.wav", wav_bytes, "audio/wav"),
            },
            timeout=60,
        )
        response.raise_for_status()
        transcribed_text = (response.json().get("transcript") or "").strip()

        if transcribed_text:
            print(f"Transcription successful: {transcribed_text}")
            return (transcribed_text, None)

        return (None, "Could not understand the audio. Please speak clearly and try again.")

    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        print(f"Meta transcription HTTP error: {e}")
        if status == 429:
            return (None, "Meta speech recognition is rate limited. Please retry in a moment.")
        return (None, f"Meta speech recognition failed (HTTP {status}).")
    except requests.RequestException as e:
        print(f"Meta transcription request error: {e}")
        return (None, "Could not connect to Meta speech recognition service. Please check your internet connection.")
    except Exception as e:
        print(f"Error transcribing audio with Meta: {e}")
        return (None, f"Transcription error: {str(e)}")


def transcribe_audio_file(audio_file_path: str, voice_id: str = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribe an audio file to text using Meta Muse Voice Transcribe.
    
    Args:
        audio_file_path: Path to the audio file
        voice_id: Optional voice ID
    
    Returns:
        tuple: (transcribed_text, error_message) - transcribed_text is the text if successful,
               error_message contains details if transcription fails. One of them will be None.
    """
    try:
        with open(audio_file_path, "rb") as f:
            audio_data = f.read()
        return transcribe_audio(audio_data, voice_id)
    except IOError as e:
        error_msg = f"Error reading audio file: {e}"
        print(error_msg)
        return (None, error_msg)