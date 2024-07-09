from django.urls import path
# urls.py or wherever you import views
from .views import UploadImageView, TrainModelView


urlpatterns = [
    path('upload/', UploadImageView.as_view(), name='upload_image'),
    path('api/train/', TrainModelView.as_view(), name='train_model'),
]
