import os
from rest_framework import generics, status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from .models import UploadedImage
from .serializers import UploadedImageSerializer
from .utils import predict_image, train_model  # Import predict_image and train_model functions

class UploadImageView(generics.CreateAPIView):
    queryset = UploadedImage.objects.all()
    serializer_class = UploadedImageSerializer
    parser_classes = (MultiPartParser, FormParser)

    def perform_create(self, serializer):
        image_instance = serializer.save()

        # Predict using the uploaded image
        predicted_class, confidence = predict_image(image_instance.image.path)

        # Update the instance with predictions
        image_instance.predicted_class = predicted_class
        image_instance.confidence = confidence
        image_instance.save()

        # Construct the response data
        response_data = {
            "id": image_instance.id,
            "image": image_instance.image.url,
            "predicted_class": predicted_class,
            "confidence": confidence
        }

        return Response(response_data, status=status.HTTP_201_CREATED)

class TrainModelView(generics.GenericAPIView):
    def get(self, request, *args, **kwargs):
        try:
            train_model()  # Train the model
            return Response({"message": "Model trained successfully"}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
