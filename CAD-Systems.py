import pandas as pd
import numpy as np
from transform import SingularDataframe
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skimage import io, color, feature, transform
from tensorflow.keras import layers, models

df = SingularDataframe()
v,t = df.df_paths_adjusted

def preprocess_image(image_path):
    img = io.imread(image_path, as_gray=True)

    img_resized = transform.resize(img, (100, 100))

    img_canny = feature.canny(img_resized)

    return img_canny

X = np.array([preprocess_image(image_path) for image_path in t['FilePath']])
y = np.array(t['Label'])

Xv = np.array([preprocess_image(image_path) for image_path in v['FilePath']])
yv = np.array(v['Label'])


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
yv_encoded = label_encoder.transform(yv)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

X_val, Xv_test, y_val, yv_test = train_test_split(Xv, yv_encoded, test_size=0.2, random_state=42)

# Reshape the data for compatibility with Conv2D layer
X_train = X_train.reshape(-1, 100, 100, 1)
X_test = X_test.reshape(-1, 100, 100, 1)
X_val = X_val.reshape(-1, 100, 100, 1)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

train_loss, train_acc = model.evaluate(X_train, y_train)
print(f'Train Accuracy: {train_acc}, Train Loss: {train_loss}')

model.save('mura_fracture_detection_model.h5')