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