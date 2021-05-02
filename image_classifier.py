import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

#load data set
fashion_data = keras.datasets.fashion_mnist

#pull out data from data set
(train_images, train_labels), (test_images, test_labels) = fashion_data.load_data()

#defining the neural network structure (sequantial process)
model = keras.Sequential([

    #input is a 28x28 pixel image (flatten means that the 28x28 is gonna be converted into one 748x1 input layer (one column))
    keras.layers.Flatten(input_shape=(28,28)),

    #hidden layer is 128 deep (128 rows and 1 column). Relu return the value, or 0 (works good enough, much faster)
    keras.layers.Dense(128, activation=tf.nn.relu),

    #output is 0-10 (depending on what piece of clothing it is) return the max (best probability)
    keras.layers.Dense(10, activation=tf.nn.softmax)

])

#compile defined above model
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train the model, using the data set
model.fit(train_images, train_labels, epochs = 5)

test_loss = model.evaluate(test_images, test_labels)

predictions = model.predict(test_images)

print(predictions[1])

#show data using matplotlib
plt.imshow(train_images[1], cmap='gray', vmin=0, vmax=255)
plt.show()