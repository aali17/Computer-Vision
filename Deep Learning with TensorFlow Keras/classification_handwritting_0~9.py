import tensorflow as tf
from os import path, getcwd, chdir

path = f"{getcwd()}/../tmp2/mnist.npz"

# GRADED FUNCTION: train_mnist
def train_mnist():

    mnist = tf.keras.datasets.mnist

    (training_images, training_labels),(test_images, test_labels) = mnist.load_data(path=path)
    training_images = training_images/255.0
    test_images = test_images/255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    model.fit(training_images, training_labels, epochs=5)
    model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)
    return 0

train_mnist()
