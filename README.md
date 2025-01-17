
# Django_Classifier Project
## Overview

CatDog Classifier is a Django-based web application that uses machine learning models to classify images as either cats or dogs. This project utilizes the Django REST Framework to handle image uploads, and TensorFlow/Keras for building and training deep learning models.

# Features
- Upload images of cats or dogs to get a predicted classification with confidence scores.
- Train the classification models on new datasets directly through the API.
- View API documentation via Swagger and Redoc.


# Installation
## Prerequisites
- Python 3.8+
- Django 4.2.13
- PostgreSQL 
- TensorFlow/Keras
- pipenv (for virtual environment management)

1. Clone the repository:

```bash
  git clone https://github.com/Vikasreddyofficial/Django_Classification.git
  cd Django_Classification
```
2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate 

```
3. Install Dependencies

```bash
pip install -r requirements.txt

```
4. Set up the PostgreSQL database:

- Ensure you have PostgreSQL installed and running.

5. Apply the migrations:

```bash
python manage.py migrate

```
6. Run the development server:

```bash
python manage.py runserver

```

## Dataset Structure
``` bash
media/
├── traind/                    # Training images for dogs
│   ├── dogs/
│   │   ├── dog1.jpg
│   │   ├── dog2.jpg
│   │   └── ...
│
├── validationd/               # Validation images for dogs
│   ├── dogs/
│   │   ├── dog1.jpg
│   │   ├── dog2.jpg
│   │   └── ...
│
├── trainc/                    # Training images for cats
│   ├── cats/
│   │   ├── cat1.jpg
│   │   ├── cat2.jpg
│   │   └── ...
│
└── validationc/               # Validation images for cats
    ├── cats/
    │   ├── cat1.jpg
    │   ├── cat2.jpg
    │   └── ...


```

# Configuration

## API Documentation:

- Swagger UI: http://127.0.0.1:8000/swagger/
- ReDoc UI: http://127.0.0.1:8000/redoc/


## API Endpoints
Here's a list of available API endpoints in the project:

1. Upload Image:

- Endpoint: /api/upload/
- Method: POST
- Description: Upload an image for classification.
- Request Body:
   - image: Image file to upload.

2. Train Model

- Endpoint: /api/train/
- Method: GET
- Description: Initiate training of the cat and dog classification models.


## Dependencies
The project requires the following packages:
- Django==4.2.13
- djangorestframework
- drf_yasg
- psycopg2-binary # PostgreSQL adapter for Python
For the full list of dependencies, refer to the requirements.txt file.

## Contributing
Contributions are welcome! Fork the repository, make your changes, and submit a pull request.

## Acknowledgements
- Django - The web framework used.
- Django REST Framework - For building the API.
- TensorFlow - For machine learning model training.
- drf-yasg - For generating API documentation.
## License
This project is licensed under the MIT License - see the LICENSE file for details