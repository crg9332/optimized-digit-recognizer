# Train and save a model to regonize handwritten digits
import tensorflow as tf # Import tensorflow library
import matplotlib.pyplot as plt # Import matplotlib library
import numpy as np
from PIL import Image

mnist = tf.keras.datasets.mnist # Object of the MNIST dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data() # Load data

# plt.imshow(x_train[0], cmap="gray") # Import the image
# plt.show() # Plot the image

# Normalize the train dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)
# Normalize the test dataset
x_test = tf.keras.utils.normalize(x_test, axis=1)

#Build the model object
model = tf.keras.models.Sequential()
# Add the Flatten Layer
model.add(tf.keras.layers.Flatten())
# Build the input and the hidden layers
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# Build the output layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x=x_train, y=y_train, epochs=5) # Start training process

# Evaluate the model performance
test_loss, test_acc = model.evaluate(x=x_test, y=y_test)
# Print out the model accuracy
print('\nTest accuracy:', test_acc)

predictions = model.predict([x_test]) # Make prediction

# Save the model and weights
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

while True:
    index = int(input("Give an index (-1 to quit): "))
    if index == -1:
        break
    print("Guess:", np.argmax(predictions[index])) # Print out the predicted number

    plt.imshow(x_test[index], cmap="gray") # Import the image
    plt.show() # Show the image
