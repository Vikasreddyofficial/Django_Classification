import os
import numpy as np
import tensorflow as tf
from keras.src.models import model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


# Constants
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2  # dog, cat

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'media', 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'media', 'validation')
MODEL_PATH = os.path.join(BASE_DIR, 'catdog_classifier_model.h5')

# Create model architecture
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# Train the model
def train_model():
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    model = create_model()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Define the checkpoint path with .weights.h5 extension
    checkpoint_path = "training/cp.weights.h5"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Configure ModelCheckpoint to save weights only
    cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                  save_weights_only=True,
                                  verbose=1)

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // BATCH_SIZE,
        callbacks=[cp_callback]
    )

    # Save the entire model to a HDF5 file (optional)
    model.save(MODEL_PATH)

    return model  # Return the trained model object

# Function to predict an image
model = load_model(MODEL_PATH)

def predict_image(image_path):
    try:
        print(f"Predicting image: {image_path}")  # Log statement to print image path

        # Load and preprocess the image
        img = image.load_img(image_path, target_size=IMAGE_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0  # Normalize pixel values

        # Make predictions
        classes = model.predict(x)
        predicted_class = "cat" if classes[0] < 0.5 else "dog"
        confidence = float(classes[0])  # Assuming binary classification

        return predicted_class, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return "", None