import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 128

model = load_model("model.h5")

image = cv2.imread("test.jpg")

image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image = image.astype("float32") / 255.0
image = np.expand_dims(image, axis=0)

prediction = model.predict(image)

classes = ["Fake Currency", "Real Currency"]

result = classes[np.argmax(prediction)]

print("Prediction :", result)
print("Confidence :", np.max(prediction) * 100, "%")