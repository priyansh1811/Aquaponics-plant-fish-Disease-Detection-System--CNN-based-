import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ✅ Corrected dataset path
data_dir = "/Users/priyansh18/Downloads/plant/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"

# Define image size & batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# ✅ Data Augmentation and Loading
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
 
# Load training and validation datasets
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ✅ Print number of detected classes before training
print(f"Detected {train_generator.num_classes} classes in training data")  

# ✅ Build CNN Model with Explicit 38 Output Classes
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
    Dense(38, activation='softmax')  # ✅ Explicitly set 38 classes
])

# ✅ Compile Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train Model
EPOCHS = 20
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ✅ Save Model with New Name
model.save("plant_disease_detection_cnn_fixed.h5")

# ✅ Plot Accuracy & Loss
def plot_metrics(history):
    plt.figure(figsize=(12,5))
    
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    
    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    
    plt.show()

plot_metrics(history)

# ✅ Verify Model Output Shape
print("Final Model Output Shape:", model.output_shape)  # Should be (None, 38)