import keras.models as models
import keras.layers as layers
from enum import Enum

class PreprocessorOption(Enum):
    singleDimension = 1
    doubleDimension = 2

WIDTH = 173
HEIGHT = 40
INPUTSHAPE = (HEIGHT, WIDTH, 1)
AUDIO_PATH = "UrbanSound8K/audio/"
TEST_PATH = "test/"

CATEGORIES = ["Air Conditioner",
                "Car Horn",
                "Children Playing",
                "Dog bark",
                "Drilling",
                "Engine Idling",
                "Gun shot",
                "Jackhammer",
                "Siren",
                "Street Music"]


FULLYCONNECTED_MODEL = models.Sequential([
    layers.Dense(500, input_shape=(HEIGHT,), activation='relu'),
    layers.Dense(250, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

# 94% 92%
CONVOLUTIONAL_MODEL1 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='valid', input_shape=INPUTSHAPE),
    layers.MaxPooling2D(2, padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='valid'),
    layers.MaxPooling2D(2, padding='same'),
    layers.Dropout(0.3),
    layers.Conv2D(128, (3, 3), activation='relu', padding='valid'),
    layers.MaxPooling2D(2, padding='same'),
    layers.Dropout(0.3),
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


#80% 81%
CONVOLUTIONAL_MODEL2 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUTSHAPE),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 91% 91%
CONVOLUTIONAL_MODEL3 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='valid', input_shape=INPUTSHAPE),
    layers.MaxPooling2D(2, padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='valid'),
    layers.MaxPooling2D(2, padding='same'),
    layers.Dropout(0.3),
    layers.Conv2D(128, (3, 3), activation='relu', padding='valid'),
    layers.MaxPooling2D(2, padding='same'),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])


