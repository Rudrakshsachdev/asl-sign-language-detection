import numpy as np # Importing numpy for handling numerical operations
from tensorflow.keras.models import Sequential # Creating a sequential cnn model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # Layers for the cnn model
from tensorflow.keras.optimizers import Adam # optimizer for training the model
from Preprocessing import load_dataset, split_data # Importing custom preprocessing functions

# STEP 1: Loading and preprocessing the dataset...

print(f"Loading and preprocessing the dataset....")
X, y = load_dataset("../data/rawData/asl_alphabet_train/asl_alphabet_train", img_size=64) # Loading images and resizing them to 64x64
X_train, X_test, y_train, y_test = split_data(X, y) # Splitting the dataset into training and testing datasets

# STEP 2: Building the CNN Model
print("Building the CNN model....")
model = Sequential()  #Initialising a sequential model


# First convolutional layer of cnn with 32 different features of size 3x3 with 'relu' activation
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(64, 64, 3))) # input shape is 64x64 with 3 color channels
model.add(MaxPooling2D(pool_size=(2,2))) # shrinking the image by 2x2 (64x64 -> 32x32)

# Second Convolutional layer of cnn with 64 filters or different features of size 3x3 with 'relu' activation
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2))) # Shrinking the image by 2x2 (32x32 -> 16x16)

model.add(Flatten()) # Converts 3D features into 1D features for featuring into dense layers

# Dense layers with 128 neurons
model.add(Dense(128, activation='relu')) # Adding a dense layer for learning complex patterns

model.add(Dropout(0.5)) # During training of a model, 50% of the neurons gets dropped randomly so for preventing overfitting we use this

# output layer with 29 neurons one for each class (A-Z)
model.add(Dense(29, activation='softmax')) # softmax gives probability distribution over classes


# STEP 3: Compile the model

model.compile(
    optimizer=Adam(learning_rate=0.001),  # Using adam optimizer with learning rate of 0.001
    loss="categorical_crossentropy",  # Loss function for multi class classification
    metrics=['accuracy'] # Monitoring accuracy during training
)

# STEP 4: Train the model

print("Training the CNN model....")
history = model.fit(
    X_train, y_train,  # Training the data and labels
    epochs=10, # Number of complete passes through the data
    batch_size=64, # Number of images processed at once
    validation_data=(X_test, y_test) # Validating data to check performance during training
)

# STEP 5: Save the model
model.save("../models/asl_model.h5") # saved the trained model to .h5 file
print("Model saved successfully....")