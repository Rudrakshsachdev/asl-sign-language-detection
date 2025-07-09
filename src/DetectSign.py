import os # for handling file path operations
import random # Importing this module for randomly selecting items from a list.
import numpy as np # for handling numerical operations
import matplotlib.pyplot as plt # for displaying results visually
import cv2 # OpenCv for immage processing
from tensorflow.keras.models import load_model # importing this for loading the pre trained model

# Loading the model
model = load_model("../models/asl_model.h5") # loading the trained model from the specified location

# Setting the image parameters
img_size = 64  # defining the image size (64x64) expected by the model
dataset_path = "../data/rawData/asl_alphabet_train/asl_alphabet_train" # Path to the dataset for fetching out the labelled datasets
labels = sorted(os.listdir(dataset_path)) # Getting all the subfolders name in sorted order


# Picking up a random label folder
random_label = random.choice(labels)
label_path = os.path.join(dataset_path, random_label) # Construct the path to the labels image folder

# Ensuring that the current path is valid
while not os.path.isdir(label_path):
    random_label = random.choice(labels)  # Keep checking out the folder until found the valid
    label_path = os.path.join(dataset_path, random_label)


# picking up a random image from label folders
random_file_image = random.choice(os.listdir(label_path)) # pick a random image from the specified path
img_path = os.path.join(label_path, random_file_image) # Full path to the image file


# Load and Preprocess the image
img = cv2.imread(img_path)
img = cv2.resize(img, (img_size, img_size)) # Resizing the image to the expected size by the model
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting from bgr to rgb format
img_normalized = img / 255.0 # Normalized pixels
img_reshaped = np.reshape(img_normalized, (1, img_size, img_size, 3)) # Reshaping the imag for batch dimensions


# Making Predictions
predictions = model.predict(img_reshaped) # Predicting class probabilities using model
predicted_index = np.argmax(predictions) # Getting the index of the class with the highest Probability
predicted_label = labels[predicted_index] # Map the index to actual label

# Display the image and prediction
plt.imshow(img_rgb)
plt.title(f"Prediction Sign: {predicted_label}")
plt.axis("off")
plt.show()
