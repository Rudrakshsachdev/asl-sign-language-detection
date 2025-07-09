import os  # for handling file path operations
import numpy as np # for handling numerical operations
from keras.src.metrics.accuracy_metrics import accuracy
from tensorflow.keras.models import load_model # importing the trained model
from Preprocessing import load_dataset, split_data # Importing custom function for loading and splitting of data

# STEP 1: Load and Preprocess the dataset

print("Loading data....")

# importing the full data (images + labels)
X, y = load_dataset("../data/rawData/asl_alphabet_train/asl_alphabet_train", img_size=64)

_, X_test, _, y_test = split_data(X, y) # Splitting into test and train set (here, ony keeping the test one)

# STEP 2: Load the trained model

print("Loading model....")

model = load_model("../models/asl_model.h5")


# STEP 3: Evaluating the model using some metrics

print("Evaluating the model....")

loss, accuracy = model.evaluate(X_test, y_test) # Evaluating the model using test data and returning the loss and accuracy

# STEP 4: Printing the results

print(f"Test Accuracy: {accuracy * 100:.2f}%") # converting the result to percentage
print(f"Test loss: {loss:.4f}%") # rounding off to 4th decimal