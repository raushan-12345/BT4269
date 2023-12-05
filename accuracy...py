#!/usr/bin/env python
# coding: utf-8

# # Object Detection using Deep Learning

# # 1. Importing the libraries

# In[1]:


import numpy as np
import cv2
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set the random seed for reproducibility
np.random.seed(42)


# # 2.Define the Dataset

# In[2]:


# Define the path to the dataset directory
dataset_dir = "C:\\Users\\raush\\Downloads\\object detection\\object detection\\Dataset"


# Define the class labels
class_labels = ['car', 'Bus', 'motor cycle', 'Truck', 'Auto', 'Tempo Traveller','Tractor']
num_classes = len(class_labels)

# Define the input image dimensions
input_shape = (150, 150, 3)
class_folders=os.listdir(dataset_dir)


# # 3. Model Building

# In[3]:


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# # 4. image processing

# In[4]:



# Initialize empty lists for storing the images and labels
images = []
labels = []

# Load and preprocess the dataset
for class_label in class_labels:
    if class_label in class_folders:
        class_dir = os.path.join(dataset_dir, class_label)
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, input_shape[:2])
            image = image.astype('float32') / 255.0
            images.append(image)
            labels.append(class_labels.index(class_label))
    else:
        print(f"Folder not found for class label: {class_label}")

# Convert the images and labels to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Convert the labels to one-hot encoding
labels = np.eye(num_classes)[labels]


# In[5]:


# Split the dataset into training and testing sets (80% for training, 20% for testing)
split = int(0.8 * len(images))
train_images, test_images = images[:split], images[split:]
train_labels, test_labels = labels[:split], labels[split:]


# # 5. Train the model and find the accuracy

# In[6]:


# Train the model
model.fit(train_images, train_labels, batch_size=32, epochs=10, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=1)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# In[ ]:




