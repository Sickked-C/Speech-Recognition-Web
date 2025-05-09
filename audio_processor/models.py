from django.db import models

class Transcription(models.Model):
    text = models.TextField()  # Lưu trữ văn bản đã nhận dạng
    language = models.CharField(max_length=2)  # Lưu trữ ngôn ngữ (vi, en)
    created_at = models.DateTimeField(auto_now_add=True)  # Thời gian tạo bản ghi

    def __str__(self):
        return f"{self.text} ({self.language})"
 
 #kết nối database   
class AudioFile(models.Model):
    recorded_file_path = models.FileField(upload_to='recorded_audio/', blank=True, null=True)
    file = models.FileField(upload_to='audio_files/',blank=True, null=True)
    language = models.CharField(max_length=10)
    transcription = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.file.name} - {self.language}'
    