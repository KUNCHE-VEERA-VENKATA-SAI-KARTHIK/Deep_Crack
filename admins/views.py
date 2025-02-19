from django.shortcuts import render,redirect
from mains.models import*
from users.models import*
from admins.models import *
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import HttpResponse
import os
import shutil
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from django.contrib import messages
from keras.models import load_model
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # Assuming 'SVM','KNN' is your model in models.py to store the metrics

#gradient boost machine algo for getting acc ,precession , recall , f1 score
# Create your views here.
def adminlogout(req):
    messages.info(req,'You are logged out...!')
    return redirect('index')

def admindashboard(req):
    return render(req,'admin/admin-dashboard.html')


from django.shortcuts import render
from django.http import HttpResponse
from .models import rfcn_b
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import json

# Custom Bilinear Interpolation Layer
class BilinearInterpolationLayer(Layer):
    def __init__(self, target_size=(7, 7), **kwargs):
        super(BilinearInterpolationLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size, method='bilinear')


def rfcn_b_alg(request):
    try:
        # Path to dataset
        dataset_dir = 'D:/Users/kunch/Desktop/Deep Crack/deep_crack/dataset'  # Adjust this path as needed

        # Data preparation
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

        # Training and validation data generators
        train_generator = datagen.flow_from_directory(
            dataset_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='training'
        )

        validation_generator = datagen.flow_from_directory(
            dataset_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )

        # Load ResNet50 base model
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False

        # Add custom layers
        x = base_model.output
        x = BilinearInterpolationLayer(target_size=(7, 7))(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)

        # Compile the model
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(train_generator, epochs=5, validation_data=validation_generator)

        # Save the model
        model.save('rfcn_b_model.h5')

        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(validation_generator)

        # Generate predictions and compute metrics
        Y_pred = model.predict(validation_generator)
        Y_pred_classes = np.round(Y_pred).flatten()
        Y_true = validation_generator.classes

        accuracy = accuracy_score(Y_true, Y_pred_classes)
        precision = precision_score(Y_true, Y_pred_classes)
        recall = recall_score(Y_true, Y_pred_classes)
        f1 = f1_score(Y_true, Y_pred_classes)
        class_report = classification_report(Y_true, Y_pred_classes, target_names=validation_generator.class_indices.keys())
        conf_matrix = confusion_matrix(Y_true, Y_pred_classes)

        # Store metrics in the database
        metrics = rfcn_b.objects.create(
            name="ResNet50 with Bilinear Interpolation",
            validation_loss=val_loss,
            validation_accuracy=val_accuracy,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            classification_report=class_report,
            confusion_matrix=json.dumps(conf_matrix.tolist())
        )

        # Prepare data for rendering
        context = {
            'name': metrics.name,
            'validation_loss': round(metrics.validation_loss, 2),
            'validation_accuracy': round(metrics.validation_accuracy, 2),
            'accuracy': round(metrics.accuracy, 2),
            'precision': round(metrics.precision, 2),
            'recall': round(metrics.recall, 2),
            'f1_score': round(metrics.f1_score, 2),
            'classification_report': metrics.classification_report,
            'confusion_matrix': json.loads(metrics.confusion_matrix),
        }

        return render(request, 'admin/RFCN_B_alg.html', context)

    except Exception as e:
        return HttpResponse(f"An error occurred: {e}", status=500)



