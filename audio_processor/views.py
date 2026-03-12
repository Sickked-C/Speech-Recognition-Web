import os
import torch
import numpy as np
import librosa
import noisereduce as nr
import sounddevice as sd
import soundfile as sf

from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment
from pydub.silence import split_on_silence

from .models import AudioFile
from .mongo import get_db
from datetime import datetime
# ─────────────────────────────────────────
# Constants
# ─────────────────────────────────────────
DURATION = 10    # Recording duration in seconds
SAMPLE_RATE = 16000

MODEL_CONFIG = {
    'vi': "khanhld/wav2vec2-base-vietnamese-160h",
    'en': "jonatasgrosman/wav2vec2-large-xlsr-53-english"
}

# ─────────────────────────────────────────
# 1. Lazy model loading (load once, reuse)
# ─────────────────────────────────────────
_MODELS = {}

def get_models():
    """Load models on first call, then reuse from cache."""
    if not _MODELS:
        print("Loading models... (this may take a moment)")
        for lang, model_name in MODEL_CONFIG.items():
            _MODELS[f'processor_{lang}'] = Wav2Vec2Processor.from_pretrained(model_name)
            _MODELS[f'model_{lang}'] = Wav2Vec2ForCTC.from_pretrained(model_name)
        print("Models loaded successfully.")
    return _MODELS

# ─────────────────────────────────────────
# 2. Record audio from microphone
# ─────────────────────────────────────────
def record_voice():
    """Record audio and save to MEDIA_ROOT."""
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    # Tự tạo thư mục nếu chưa có
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

    file_name = "recorded_audio.wav"
    file_path = os.path.join(settings.MEDIA_ROOT, file_name)
    sf.write(file_path, audio, SAMPLE_RATE)

    file_url = os.path.join(settings.MEDIA_URL, file_name)
    return file_url, file_path

# ─────────────────────────────────────────
# 3. Audio preprocessing
# ─────────────────────────────────────────
def reduce_noise(audio, sr):
    """Reduce background noise using noisereduce."""
    return nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.85)

def remove_silence(file_path):
    """Remove silent parts from audio file."""
    sound = AudioSegment.from_wav(file_path)
    chunks = split_on_silence(
        sound,
        min_silence_len=250,
        silence_thresh=sound.dBFS - 0.9,
        keep_silence=200
    )
    non_silent_audio = AudioSegment.empty()
    for chunk in chunks:
        non_silent_audio += chunk
    return non_silent_audio

def audiosegment_to_numpy(audio_segment):
    """Convert pydub AudioSegment to normalized numpy array."""
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    if audio_segment.sample_width == 2:  # 16-bit PCM
        samples /= 32768.0
    return samples

# ─────────────────────────────────────────
# 4. Speech transcription
# ─────────────────────────────────────────
def transcribe_audio(audio, sr, language='vi'):
    """Run Wav2Vec2 inference on audio array."""
    models = get_models()
    processor = models[f'processor_{language}']
    model = models[f'model_{language}']

    input_values = processor(
        audio,
        return_tensors="pt",
        sampling_rate=sr
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]

# ─────────────────────────────────────────
# 5. Full audio processing pipeline
# ─────────────────────────────────────────
def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 file to WAV format."""
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    return wav_path

def process_uploaded_file(file_path, language='vi'):
    """
    Full pipeline:
    MP3→WAV (if needed) → remove silence → noise reduction → transcribe
    Returns (transcription, processed_audio) or (None, None) on error.
    """
    try:
        # Convert MP3 to WAV if needed
        if file_path.endswith('.mp3'):
            wav_path = file_path.replace('.mp3', '.wav')
            convert_mp3_to_wav(file_path, wav_path)
            file_path = wav_path

        # Remove silence
        non_silent_audio = remove_silence(file_path)

        # Convert to numpy
        audio_array = audiosegment_to_numpy(non_silent_audio)
        sr = non_silent_audio.frame_rate

        # Resample to 16000 Hz if needed
        if sr != SAMPLE_RATE:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sr,
                target_sr=SAMPLE_RATE
            )
            sr = SAMPLE_RATE

        # Noise reduction
        audio_array = reduce_noise(audio_array, sr)

        # Transcribe
        transcription = transcribe_audio(audio_array, sr, language)
        return transcription, non_silent_audio

    except Exception as e:
        print(f"[ERROR] process_uploaded_file: {e}")
        return None, None

def save_processed_audio(audio_segment, filename):
    """Save processed audio to MEDIA_ROOT and return its URL."""
    directory = os.path.join(settings.MEDIA_ROOT, 'processed_audio')
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, f'{filename}.wav')
    audio_segment.export(file_path, format='wav')

    file_url = os.path.join(settings.MEDIA_URL, 'processed_audio', f'{filename}.wav')
    return file_url

# ─────────────────────────────────────────
# 6. Django Views
# ─────────────────────────────────────────
def index(request):
    """Home page."""
    return render(request, 'index.html')

def process_audio_view(request):
    """Handle live recording and transcription."""
    if request.method == 'POST':
        language = request.POST.get('language', 'vi')

        file_url, file_path = record_voice()
        transcription, non_silent_audio = process_uploaded_file(file_path, language)

        if transcription is None:
            return render(request, 'index.html', {
                'error': 'Could not process audio. Please try again.'
            })

        processed_file_url = save_processed_audio(non_silent_audio, 'processed_recorded_audio')

        db = get_db()
        db.audio_files.insert_one({
            'language': language,
            'recorded_file_path': file_url,  
            'transcription': transcription,
            'created_at': datetime.now().strftime('%d/%m/%Y %H:%M:%S')    
        })

        return render(request, 'result.html', {
            'transcription': transcription,
            'audio_url': processed_file_url
        })

    return render(request, 'index.html')

def process_audio_file_view(request):
    """Handle uploaded audio file and transcription."""
    if request.method == 'POST' and request.FILES.get('audio_file'):
        language = request.POST.get('language', 'vi')
        audio_file = request.FILES['audio_file']

        file_name = default_storage.save(audio_file.name, audio_file)
        file_path = default_storage.path(file_name)

        transcription, non_silent_audio = process_uploaded_file(file_path, language)

        if transcription is None:
            return render(request, 'index.html', {
                'error': 'Could not process audio. Please try again.'
            })

        db = get_db()
        db.audio_files.insert_one({
            'language': language,
            'recorded_file_path': file_path,  
            'transcription': transcription,
            'created_at': datetime.now().strftime('%d/%m/%Y %H:%M:%S')    
        })

        file_url = save_processed_audio(non_silent_audio, 'processed_file')
        return render(request, 'result.html', {
            'transcription': transcription,
            'audio_url': file_url
        })

    return render(request, 'index.html')

def history(request):
    """Show transcription history from MongoDB."""
    db = get_db()
    records = list(db.audio_files.find(
        {},
        {'_id': 0}  
    ).sort('created_at', -1)) 
    
    return render(request, 'history.html', {'history': records})