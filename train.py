import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 128

df = pd.read_csv("Dataset.csv")

images = []
labels = []

for i in range(len(df)):
    path = df.iloc[i]["image_path"]
    label = int(df.iloc[i]["label"])

    if os.path.exists(path):
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(label)

X = np.array(images) / 255.0
y = to_categorical(labels, 2)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = Sequential()

model.add(Conv2D(32,(3,3),activation="relu",input_shape=(128,128,3)))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(128,activation="relu"))
model.add(Dense(2,activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test,y_test)
)

model.save("model.h5")

print("Training Completed")