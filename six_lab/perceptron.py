import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tensorflow import keras
from keras.layers import Dense, Flatten


def perceptron():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # стандартизация входных данных
  x_train = x_train / 255
  x_test = x_test / 255

  y_train_cat = keras.utils.to_categorical(y_train, 10)
  y_test_cat = keras.utils.to_categorical(y_test, 10)


  model = keras.Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
  ])

  print(model.summary())  # вывод структуры НС в консоль

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(x_train, y_train_cat, batch_size=20, epochs=6, validation_split=0.2)

  model.evaluate(x_test, y_test_cat)

  n = 1
  x = np.expand_dims(x_test[n], axis=0)
  res = model.predict(x)
  print(res)
  print(np.argmax(res))

  plt.imshow(x_test[n], cmap=plt.cm.binary)
  plt.show()






perceptron()
