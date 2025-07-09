import os # importing the os module for file path operations
import numpy as np # importing the numpy module for handling numerical operations
import cv2 # importing cv2 module for image loading and resizing
from sklearn.model_selection import train_test_split # for splitting the data into training and testing part
from tensorflow.keras.utils import to_categorical  # For one hot coding encoding labels

# Function to load and preprocess the dataset
def load_dataset(data_dir, img_size=64):
    """
    Loads the dataset and labels from the specified directory.
    Parameters:
         1. Data_dir: The directory where the dataset is located
         2. Img_Size: The size of the image to load

    Returns:
        - x: Numpy array of image data
        - y: Numpy array of one-hot encoded labels
    """

    x = [] # List to store the image data
    y = [] # List to store class labels

    labels = sorted(os.listdir(data_dir)) # Getting the folders name in a sorted order
    print(f"Founded the labels...{labels}")

    # Loop through each label folder
    for idx, label in enumerate(labels):
        label_pth = os.path.join(data_dir, label) # creating a full path to the folder

        # Skip if its not a folder
        if not os.path.isdir(label_pth):
            continue

        print(f"Loading images from label..{label}")

        # looping out through each image inside the folder
        for img_file in os.listdir(label_pth):
            img_path = os.path.join(label_pth, img_file) # full image path

            try:
                image = cv2.imread(img_path) # Reading the image
                image = cv2.resize(image, (img_size, img_size)) # Resizing the image to img_size * img_size
                image = image / 255.0 # Normalize pixel values

                x.append(image) # Adding the image to feature list
                y.append(idx) # Adding the corresponding class

            except Exception as e:
                print(f"Error loading image: {img_file}: {e}")

    # Converting list to numpy arrays
    x = np.array(x)
    y = np.array(y)

    # One hot encode the labels
    y = to_categorical(y, num_classes=len(labels))
    print(f"Loaded...{len(x)} images.")
    return x, y


# Function to split the dataset
def split_data(x, y, test_size=0.2):
    """
    Splits the dataset into training and test sets...
    Parameters:
    1. x: Numpy array of image data
    2. y: one-hot encoded labels
    3. test_size: fraction of the dataset to use for testing

    """

    return train_test_split(x, y, test_size=test_size, random_state=42)


if __name__ == "__main__":
    X, y = load_dataset("../data/rawData/asl_alphabet_train/asl_alphabet_train", img_size=64)
    print(f"X: {X.shape}, y: {y.shape}")




