from tensorflow.keras.models import load_model
from keras.layers import Input, Embedding, LSTM, Dropout, Dense, GlobalAveragePooling2D, Concatenate, Dot, Lambda
from keras.applications import DenseNet121
from keras.models import Model
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape
from keras.models import Model
import numpy as np
import random
import os
import cv2
import tensorflow as tf




def create_image_encoder():
    model = keras.Sequential([
    tf.keras.applications.MobileNet(include_top=False,input_shape=(100, 100, 3), weights='imagenet'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(10, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(10, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(10, (3, 3), activation='relu', padding='same'), # add another Conv2D layer
    Conv2D(2, (3, 3), activation='relu', padding='same'),
    Flatten()

    ])
    return model


# def load_image_encoder_weights(model,weight_path):
#     model.load_weights(weight_path)
