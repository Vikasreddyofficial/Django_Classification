from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')
    predicted_class = models.CharField(max_length=10, blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
