import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 128

model = load_model("model.h5")

img = cv2.imread("test.jpg")
img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

img = img/255.0
img = np.expand_dims(img,axis=0)

prediction = model.predict(img)

label = np.argmax(prediction)

if label==0:
    print("Fake Currency")
else:
    print("Real Currency")