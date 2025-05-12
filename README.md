# 🗣️ Speech-to-Text AI Web Application

This is a personal project demonstrating a simple yet effective speech-to-text web application.  
It allows users to convert spoken words into written text using either direct recording or uploaded audio files.

---

## 🔍 About the Project

This web application uses Django as the backend framework and supports both Vietnamese and English transcription using pre-trained Wav2Vec 2.0 models.

**Main goals:**
- Build an end-to-end speech recognition app
- Use Django to handle backend logic and routing
- Transcribe both recorded and uploaded audio files
- Apply HuggingFace Wav2Vec2 models for AI-powered transcription

---

## 🚀 Features
- 🎙️ Record speech directly in the browser  
- 📂 Upload `.wav` audio files  
- 🔤 Convert audio to text using Wav2Vec2 models (Vietnamese + English)  
- 📝 Display transcription results in the UI  
- 🔊 Playback the audio  
- 🌐 Language selection (EN or VI)

---

## 🛠️ Tech Stack
- **Backend:** Python, Django  
- **Frontend:** HTML, CSS, JavaScript  
- **Speech Models:**  
  - `khanhld/wav2vec2-base-vietnamese-160h` (Vietnamese)  
  - `jonatasgrosman/wav2vec2-large-xlsr-53-english` (English)  
- **Templating:** Django Templates (Jinja-like syntax)

---

## 📦 Installation

### Clone the repository
```bash
git clone https://github.com/Sickked-C/Speech-to-Text-AI-Web-Application.git
cd Speech-to-Text-AI-Web-Application

```
### Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Apply migrations
```bash
python manage.py migrate
```

### Run the app
```bash
python manage.py runserver
```
Visit http://localhost:8000 in your browser.

---

## 📁 Project Structure
```csharp
.
├── static/             # JavaScript, CSS, icons
├── templates/          # HTML templates
├── stt_project/        # Django project files
├── stt_app/            # Main app: views, urls, model inference
├── requirements.txt    # Python packages
└── README.md           # Project info
```

---

## 🎯 Future Improvements
✅ Multi-language transcription

⏳ Save transcription history to database

⏳ Add Whisper model as backend option

⏳ Improve mobile UI

⏳ Deploy to Railway or Render

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ About Me
This project is part of my personal portfolio to demonstrate:

Django web development skills

Real-time audio handling in the browser

AI-based speech recognition with HuggingFace
