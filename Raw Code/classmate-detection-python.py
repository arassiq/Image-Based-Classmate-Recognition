pip install tensorflow

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from PIL import Image

# Define directories
train_dir = '/content/train'
test_dir = '/content/test'

print('Train directory:', os.listdir(train_dir))

train_Alex_dir = os.path.join(train_dir, 'Alex')
train_Not_Alex_dir = os.path.join(train_dir, 'Not_Alex')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),
    batch_size=32,
    class_mode='binary'
)

#load test image
test_image_path = os.path.join(test_dir, 'test_image.JPG')
test_image = Image.open(test_image_path).resize((32, 32))
test_image = np.array(test_image) / 255.0
test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

#print(test_image)
#print(train_generator)

# Build simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator, epochs=10)

prediction = model.predict(test_image)

#determine whether alex is in the model
if prediction[0][0] > 0.5:
    prediction_text = "Alex detected"
    color = 'green'
else:
    prediction_text = "Alex not detected"
    color = 'red'

plt.figure(figsize=(6, 6))
plt.imshow(test_image[0])
plt.title(prediction_text, color=color)
plt.axis('off')
plt.show()