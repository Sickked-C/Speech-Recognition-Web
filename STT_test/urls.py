from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('audio_processor.urls')),  # Bao gồm các URL từ ứng dụng audio_processor
] 
