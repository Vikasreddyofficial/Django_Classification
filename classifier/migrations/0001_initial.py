# Generated by Django 4.2.13 on 2024-07-09 08:05

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='UploadedImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='images/')),
                ('predicted_class', models.CharField(blank=True, max_length=50)),
                ('confidence', models.FloatField(blank=True, null=True)),
            ],
        ),
    ]
