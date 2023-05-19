# script used to train the character recognition model

import os
import cv2
import natsort
import numpy as np
from tqdm import tqdm
from class_names import LetterList
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

print("Packages imported...\n")

print("GPU name: ", tf.config.experimental.list_physical_devices("GPU"))
print("\n")

# data_dir = (
#     os.path.abspath(os.path.join(os.pardir, "ASL_Alphabet_Dataset"))
#     + "/asl_alphabet_train/"
# )

data_dir = os.path.jpin(
    os.path.abspath(os.pardir), "ASL_Alphabet_Dataset", "asl_alphabet_train"
)

image_size = 100
num_classes = 29


def process_image(image_filename, folder_dir, folder_name, label):
    img_file = cv2.imread(
        os.path.join(folder_dir, folder_name, image_filename), cv2.IMREAD_GRAYSCALE
    )
    if img_file is not None:
        img_resized = cv2.resize(img_file, (image_size, image_size))
        img_normalized = img_resized / 255.0
        final_img = img_normalized.reshape((-1, image_size, image_size, 1))
        return final_img, label
    else:
        return None


def get_data(folder_dir):
    X = np.empty((223075, image_size, image_size, 1), dtype=np.float32)
    y = np.empty((223075), dtype=np.int32)

    counter = 0
    folders = natsort.natsorted(os.listdir(folder_dir))
    for index, folder_name in enumerate(folders):
        if not folder_name.startswith("."):
            label = index
            print(f"Processing folder: {folder_name}")
            images = natsort.natsorted(
                os.listdir(os.path.join(folder_dir, folder_name))
            )

            results = Parallel(n_jobs=-1)(
                delayed(process_image)(image_filename, folder_dir, folder_name, label)
                for image_filename in images
            )

            for res in results:
                if res is not None:
                    X[counter], y[counter] = res
                    counter += 1
            print("\n")
    return X[:counter], y[:counter]


print("Getting images from folder...\n")
X, y = get_data(data_dir)

print("\nImages successfully imported...\n")

label_dict = LetterList.getList()

print(f"\nShape of X is {X.shape}")
print(f"Shape of y is {y.shape}")
print(f"Shape of a image is {X[0].shape}\n")

print("Splitting data into train, validation, and test sets...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=True,
    random_state=50,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    train_size=0.875,
    shuffle=True,
    random_state=50,
    stratify=y_train,
)

del X
del y

print("Converted y to categorical...\n")
num_classes = 29

y_cat_train = to_categorical(y_train, num_classes)
y_cat_val = to_categorical(y_val, num_classes)
y_cat_test = to_categorical(y_test, num_classes)

del y_train
del y_val
del y_test

print("Creating model...\n")

model = Sequential()
model.add(Conv2D(32, 3, activation="relu", input_shape=(image_size, image_size, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, activation="relu"))
model.add(Conv2D(64, 3, activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(128, 3, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(84, activation="relu"))
model.add(Dense(29, activation="softmax"))

print(f"Model Summary: \n{model.summary()}\n")

adam = keras.optimizers.Adam(learning_rate=0.001)
early_stop = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

print("Compiling model...\n")
model.compile(
    optimizer=adam,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print("Training model...\n")
history = model.fit(
    X_train,
    y_cat_train,
    epochs=50,
    validation_data=(X_val, y_cat_val),
    callbacks=[early_stop],
    verbose=1,
    batch_size=64,
)

model.save("sign_2.h5")
