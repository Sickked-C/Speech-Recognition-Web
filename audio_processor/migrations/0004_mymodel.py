# Generated by Django 4.1.13 on 2024-10-23 06:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('audio_processor', '0003_transcription_delete_audiorecording'),
    ]

    operations = [
        migrations.CreateModel(
            name='MyModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
            ],
        ),
    ]
