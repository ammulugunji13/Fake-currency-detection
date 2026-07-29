# model.py - Fake Currency Training
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

print("Training Started...")

# Dataset path - neeku dataset folder name enti?
# 'Dataset' folder lo 'real' and 'fake' ane 2 folders undali
dataset_path = "Dataset"  # Leda "./counterfeit-banknote-detection/Dataset"

if not os.path.exists(dataset_path):
    print(f"ERROR: {dataset_path} folder kanapadaledu!")
    print("Dataset folder ni app.py pakkana pettu")
    exit()

# Image Generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print(f"Train Images: {train.samples}, Val Images: {val.samples}")

# Model Architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Training - 10 epochs
history = model.fit(train, validation_data=val, epochs=10)

# Save model
model.save("model.h5")
print("SUCCESS bro! model.h5 Created! Ippudu app.py run cheyochu!")