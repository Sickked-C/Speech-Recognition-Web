from django.urls import path
from .views import index,history, process_audio_view, process_audio_file_view
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', index, name='index'),  # Trang chính
    path('history/', history, name='history'),
    path('process-audio/', process_audio_view, name='process_audio'),  # Xử lý ghi âm
    path('process-file/', process_audio_file_view, name='process_audio_file'),  # Xử lý tệp tải lên
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

