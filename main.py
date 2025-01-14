import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from PIL import Image

# Set image directories
image_dir = "food-101/images"
train_file = "food-101/meta/train.txt"
test_file = "food-101/meta/test.txt"

# Categories we want to classify
target_categories = ['hamburger', 'pizza']

# Helper function to load images from txt files
def load_images_from_file(file_path, target_categories):
    images = []
    labels = []
    
    with open(file_path, 'r') as file:
        for line in file:
            category, img_name = line.strip().split('/')  # Extract category and image name
            if category in target_categories:
                img_path = os.path.join(image_dir, category, img_name + ".jpg")  # Full path to image
                images.append(img_path)
                labels.append(0 if category == 'hamburger' else 1)  # Label: 0 for hamburger, 1 for pizza
    
    return images, labels

# Load training and testing images using the helper function
train_images, train_labels = load_images_from_file(train_file, target_categories)
test_images, test_labels = load_images_from_file(test_file, target_categories)

# Function to preprocess images: resize and normalize
def preprocess_images(image_paths, target_size=(224, 224)):
    images = []
    for img_path in image_paths:
        img = Image.open(img_path)  # Open the image
        img = img.resize(target_size)  # Resize to the target size
        img_array = np.array(img) / 255.0  # Normalize the image to [0, 1]
        images.append(img_array)
    return np.array(images)

# Preprocess images for training and testing
train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

# Convert labels to numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Build the CNN model
model = Sequential()

# Convolutional layer 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))  # 32 filters, 3x3 kernel
model.add(MaxPool2D(pool_size=(2, 2)))  # Pooling layer to reduce dimensionality

# Convolutional layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

# Convolutional layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

# Flattening the output for the fully connected layer
model.add(Flatten())

# Fully connected layer
model.add(Dense(128, activation='relu'))  # Dense layer with 128 units
model.add(Dropout(0.5))  # Dropout layer to prevent overfitting

# Output layer
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification (0: hamburger, 1: pizza)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summarize the model
model.summary()

# Train the model
history = model.fit(
    train_images, train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Test the model on the test data
predictions = model.predict(test_images)

# Output the predictions for test data
for i, prediction in enumerate(predictions):
    label = 'pizza' if prediction > 0.5 else 'hamburger'
    print(f"Image {test_images[i]}: Prediction: {label}, Actual: {'pizza' if test_labels[i] == 1 else 'hamburger'}")

# Save the model
model.save('food_classifier_model.h5')