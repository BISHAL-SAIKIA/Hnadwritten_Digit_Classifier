import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test,y_test) = keras.datasets.mnist.load_data()

model = keras.Sequential(
    [keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(300,activation='relu'),
            keras.layers.Dense(200,activation='relu'),
            keras.layers.Dense(10,activation='sigmoid')
    ])
model.compile(
    optimizer = 'adam',
    loss='sparse_categorical_crossentropy',
    metrics= ['accuracy']
)
model.fit(x_train,y_train,epochs=5)
model.save('Digit_Classifier.keras')