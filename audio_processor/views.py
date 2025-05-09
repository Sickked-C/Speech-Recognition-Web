# Thư viện cần thiết
import torch
import noisereduce as nr
import librosa
from django.shortcuts import render
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import sounddevice as sd
from pydub import AudioSegment
from pydub.silence import split_on_silence
from django.core.files.storage import default_storage
import numpy as np
from .models import AudioFile
from django.conf import settings
import soundfile as sf
import os

# Định nghĩa các biến toàn cục
duration = 10  # Thời gian ghi âm (giây)
fs = 16000  # Tần số lấy mẫu âm thanh

# 1. Tải mô hình tiếng Việt và tiếng Anh
def load_models():
    global processor_vi, model_vi, processor_en, model_en
    
    processor_vi = Wav2Vec2Processor.from_pretrained("khanhld/wav2vec2-base-vietnamese-160h")
    model_vi = Wav2Vec2ForCTC.from_pretrained("khanhld/wav2vec2-base-vietnamese-160h")
    
    processor_en = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    model_en = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")

# 2. Hàm ghi âm giọng nói
def record_voice():
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Đợi đến khi ghi âm xong

    # Lưu file âm thanh đã ghi vào thư mục MEDIA_ROOT
    file_name = "recorded_audio.wav"
    file_path = os.path.join(settings.MEDIA_ROOT, file_name)

    # Lưu file dưới định dạng WAV
    sf.write(file_path, audio, fs)

    # Trả về đường dẫn URL để phát lại
    file_url = os.path.join(settings.MEDIA_URL, file_name)

    return file_url, file_path

# 3. Hàm giảm tiếng ồn bằng noisereduce
def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.85)

# 4. Hàm cắt bỏ khoảng lặng trong tệp âm thanh
def remove_silence(file_path):
    sound = AudioSegment.from_wav(file_path)
    chunks = split_on_silence(sound, 
                              min_silence_len=250,  
                              silence_thresh=sound.dBFS - 0.9,  
                              keep_silence=200)
    non_silent_audio = AudioSegment.empty()
    for chunk in chunks:
        non_silent_audio += chunk
    return non_silent_audio

# 5. Hàm nhận diện giọng nói
def transcribe_audio(audio, sr, language='vi'):
    if language == 'vi':
        processor = processor_vi
        model = model_vi
    else:
        processor = processor_en
        model = model_en

    input_values = processor(audio, return_tensors="pt", sampling_rate=sr).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

# 6. Chuyển đổi tệp âm thanh từ pydub.AudioSegment sang PyTorch Tensor
def audiosegment_to_torch_tensor(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

    # Chuẩn hóa dữ liệu về [-1, 1] nếu audio là 16-bit PCM
    if audio_segment.sample_width == 2:  # 2 bytes = 16 bits
        samples /= 32768.0  # Chia cho 2^15 để chuẩn hóa về [-1, 1]

    # Chuyển numpy array thành PyTorch tensor
    audio_tensor = torch.tensor(samples)
    
    return audio_tensor

# 7. Sử dụng hàm này trong quá trình xử lý file âm thanh
def process_uploaded_file(file_path, language='vi'):
    # Chuyển đổi tệp MP3 sang WAV nếu cần
    if file_path.endswith('.mp3'):
        wav_file_path = file_path.replace('.mp3', '.wav')
        convert_mp3_to_wav(file_path, wav_file_path)
        file_path = wav_file_path  # Cập nhật đường dẫn file thành WAV

    # Cắt bỏ khoảng lặng trong âm thanh
    non_silent_audio = remove_silence(file_path)

    # Chuyển đổi âm thanh từ AudioSegment sang PyTorch Tensor
    audio_tensor = audiosegment_to_torch_tensor(non_silent_audio)

    # Lấy tần số mẫu từ file âm thanh
    sr = non_silent_audio.frame_rate

    # Resample âm thanh về 16000 Hz nếu tần số mẫu khác 16000
    if sr != 16000:
        audio_tensor = torch.tensor(librosa.resample(audio_tensor.numpy(), orig_sr=sr, target_sr=16000))
        sr = 16000  # Cập nhật lại tần số mẫu

    # Giảm tiếng ồn
    audio_tensor = reduce_noise(audio_tensor.numpy(), sr)  # Giảm tiếng ồn yêu cầu numpy, chuyển đổi tạm thời

    # Nhận diện giọng nói
    transcription = transcribe_audio(audio_tensor, sr, language)

    return transcription, non_silent_audio

# 8. Hàm chuyển đổi file MP3 sang WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    return wav_path

# 9. Lưu âm thanh đã xử lý
def save_processed_audio(audio_segment, filename):
    directory = os.path.join(settings.MEDIA_ROOT, 'processed_audio')

    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Đường dẫn đầy đủ của file âm thanh
    file_path = os.path.join(directory, f'{filename}.wav')

    # Lưu file âm thanh đã xử lý
    audio_segment.export(file_path, format='wav')

    # Trả về đường dẫn URL của file đã lưu
    file_url = os.path.join(settings.MEDIA_URL, 'processed_audio', f'{filename}.wav')

    return file_url

# 10. View xử lý âm thanh và nhận diện giọng nói
def process_audio_view(request):
    if request.method == 'POST':
        # Lấy ngôn ngữ từ yêu cầu POST
        language = request.POST.get('language', 'vi')  # Mặc định là tiếng Việt

        # Ghi âm và lưu file
        file_url, file_path = record_voice()

        # Xử lý file âm thanh và nhận diện giọng nói
        transcription, non_silent_audio = process_uploaded_file(file_path, language)

        # Lưu âm thanh đã xử lý để phát lại
        processed_file_url = save_processed_audio(non_silent_audio, 'processed_recorded_audio')

        # Lưu thông tin vào MongoDB
        audio_record = AudioFile(
            language=language,
            recorded_file_path=file_url,  # Lưu đường dẫn file ghi âm
            transcription=transcription  # Lưu kết quả nhận diện giọng nói
        )
        audio_record.save()  # Lưu vào database

        # Trả về trang kết quả với văn bản và URL file âm thanh đã ghi
        return render(request, 'result.html', {'transcription': transcription, 'audio_url': processed_file_url})

    return render(request, 'index.html')

# 11. View xử lý âm thanh tải lên
def process_audio_file_view(request):
    if request.method == 'POST' and request.FILES['audio_file']:
        
        # Lấy file được tải lên
        audio_file = request.FILES['audio_file']
        file_name = default_storage.save(audio_file.name, audio_file)
        file_path = default_storage.path(file_name)

        # Chọn ngôn ngữ để xử lý
        language = request.POST.get('language', 'vi')

        # Xử lý và nhận diện giọng nói từ file âm thanh
        transcription, non_silent_audio = process_uploaded_file(file_path, language)

        # Kiểm tra nếu transcription có giá trị hợp lệ
        if transcription is not None:
            # Lưu thông tin file vào MongoDB
            audio_record = AudioFile(file=audio_file, language=language, transcription=transcription)
            audio_record.recorded_file_path = file_path  # Cập nhật đường dẫn file ghi âm

            # Lưu vào database
            audio_record.save()

            # Lưu âm thanh đã xử lý để phát lại
            file_url = save_processed_audio(non_silent_audio, 'processed_file')

            return render(request, 'result.html', {'transcription': transcription, 'audio_url': file_url})

        else:
            # Nếu transcription là None, trả về thông báo lỗi
            return render(request, 'index.html', {'error': 'Không thể nhận diện âm thanh. Vui lòng thử lại.'})

    return render(request, 'index.html')

# 12. Trang chủ
def index(request):
    return render(request, 'index.html')

# 13. Lịch sử các bản ghi âm
def history(request):
    history = AudioFile.objects.all()  # Lấy tất cả các bản ghi từ database
    return render(request, 'history.html', {'history': history})

# 14. Load các mô hình khi ứng dụng khởi động
load_models()
