from django.shortcuts import render,redirect
from mains.models import *
from users.models import Dataset
from admins.models import *
from django.contrib import messages
import time
from django.core.paginator import Paginator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.core.files.storage import default_storage
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.shortcuts import render, redirect
from .models import User  # Ensure your models are imported
from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def userdashboard(req):
    images_count =  User.objects.all().count()
    print(images_count)
    user_id = req.session["User_id"]
    user = User.objects.get(User_id = user_id)
    return render(req,'user/user-dashboard.html')
  



def userlogout(req):
    user_id = req.session["User_id"]
    user = User.objects.get(User_id = user_id) 
    t = time.localtime()
    user.Last_Login_Time = t
    current_time = time.strftime('%H:%M:%S', t)
    user.Last_Login_Time = current_time
    current_date = time.strftime('%Y-%m-%d')
    user.Last_Login_Date = current_date
    user.save()
    messages.info(req, 'You are logged out..')
    return redirect('index')




import os
import numpy as np
from django.shortcuts import render
from django.core.files.storage import default_storage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Layer
import tensorflow as tf

# Define the custom BilinearInterpolationLayer
class BilinearInterpolationLayer(Layer):
    def __init__(self, target_size=(7, 7), **kwargs):
        super(BilinearInterpolationLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size, method='bilinear')

# Load the trained model with the custom layer
MODEL_PATH = 'D:/Users/kunch/Desktop/Deep Crack/deep_crack/rfcn_b_model.h5'
model = load_model(MODEL_PATH, custom_objects={'BilinearInterpolationLayer': BilinearInterpolationLayer})

# Class labels (update based on your dataset)
CLASS_NAMES = {0: 'Negative', 1: 'Positive'}

def predict_image(request):
    if request.method == 'POST' and request.FILES['image']:
        # Save uploaded file temporarily
        uploaded_file = request.FILES['image']
        file_path = default_storage.save(f"uploads/{uploaded_file.name}", uploaded_file)
        
        # Get the full file path
        full_file_path = default_storage.path(file_path)  # This gives the absolute file path

        try:
            # Preprocess the image
            img = load_img(full_file_path, target_size=(224, 224))  # Resize to model input size
            img_array = img_to_array(img) / 255.0  # Normalize the image
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make prediction
            prediction = model.predict(img_array)
            prediction_class = int(np.round(prediction[0][0]))
            prediction_label = CLASS_NAMES[prediction_class]
            confidence = prediction[0][0] if prediction_class == 1 else 1 - prediction[0][0]
        finally:
            # Clean up uploaded file by deleting the file at the absolute path
            os.remove(full_file_path)

        # Return result
        return render(request, 'user/predict.html', {
            'prediction': prediction_label,
            'confidence': f"{confidence * 100:.2f}%",
        })

    return render(request, 'user/predict.html')






















        





    
   

