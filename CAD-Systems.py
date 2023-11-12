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