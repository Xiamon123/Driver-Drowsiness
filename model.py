import os
from keras.preprocessing import image
import numpy as np
from keras.models import Sequential
from keras.layers import (
    Dropout,
    Conv2D,
    Flatten,
    Dense,
    MaxPooling2D,
    BatchNormalization,
)
from keras.preprocessing.image import ImageDataGenerator


def generator(
    dir,
    gen=image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    ),
    shuffle=True,
    batch_size=32,
    target_size=(24, 24),
    class_mode="categorical",
):
    return gen.flow_from_directory(
        dir,
        batch_size=batch_size,
        shuffle=shuffle,
        color_mode="grayscale",
        class_mode=class_mode,
        target_size=target_size,
    )


BS = 32
TS = (24, 24)

train_batch = generator("data/train", shuffle=True, batch_size=BS, target_size=TS)
valid_batch = generator("data/test", shuffle=True, batch_size=BS, target_size=TS)


train_steps = len(train_batch.classes) // BS
valid_steps = len(valid_batch.classes) // BS


model = Sequential(
    [
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(24, 24, 1)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu"),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(4, activation="softmax"),
    ]
)

# model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


history = model.fit(
    train_batch,
    validation_data=valid_batch,
    epochs=25,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
)

model.save("models/cnncvip3.h5", overwrite=True)

import matplotlib.pyplot as plt


def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")

    plt.tight_layout()
    plt.show()


# Assuming you have already trained your model and have the history object
# If not, you need to train the model first
# history = model.fit(...)

plot_training_history(history)
