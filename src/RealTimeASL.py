import cv2  # OpenCV for webcam and image handling
import numpy as np  # For numerical operations
from tensorflow.keras.models import load_model  # To load our trained model
import os
import pyttsx3


# --------------------------
# Load trained model
# --------------------------
model = load_model("../models/asl_model.h5")  # Load your trained model

# --------------------------
# Load labels (folder names)
# --------------------------
labels = sorted(os.listdir("../data/rawData/asl_alphabet_train/asl_alphabet_train"))  # Must match training labels
print("✅ Labels loaded:", labels)

# --------------------------
# Initialize webcam
# --------------------------
cap = cv2.VideoCapture(0)  # Use default camera

# --------------------------
# For stable predictions
# --------------------------
prev_label = ""
same_count = 0
stable_threshold = 10  # Lowered for easier testing
sentence = ""

engine = pyttsx3.init() # initialize text to speech engine


# --------------------------
# Real-time loop
# --------------------------
while True:
    ret, frame = cap.read()  # Read a frame
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # Mirror the image for user-friendly interaction

    # Define region of interest (ROI) where hand should be shown
    x1, y1, x2, y2 = 50, 50, 400, 400
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the ROI box

    # Extract and preprocess the ROI
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        continue

    cropped_img = cv2.resize(roi, (64, 64))  # Resize to match training input
    cropped_img = cropped_img / 255.0  # Normalize pixel values
    cropped_img = np.expand_dims(cropped_img, axis=0)  # Add batch dimension

    # Predict using the model
    prediction = model.predict(cropped_img)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index]

    # Add confidence threshold
    if confidence > 0.85:
        predicted_label = labels[predicted_index]
    else:
        predicted_label = "Unknown"

    # Print prediction every frame
    print(f"🔍 Predicted: {predicted_label} ({confidence * 100:.2f}%)")

    # Display the prediction above the ROI box
    cv2.putText(frame, f"{predicted_label} ({confidence * 100:.1f}%)",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Stabilize prediction before adding to sentence
    if predicted_label == prev_label:
        same_count += 1
    else:
        same_count = 0
        prev_label = predicted_label

    if same_count == stable_threshold:
        if predicted_label == "space":
            sentence += " "
        elif predicted_label == "del":
            sentence = sentence[:-1]
        elif predicted_label not in ["nothing", "Unknown"]:
            sentence += predicted_label

        same_count = 0
        print("📝 Current sentence:", sentence)

    # Display the full sentence at the bottom of the screen
    cv2.putText(frame, f"Sentence: {sentence if sentence else '...'}",
                (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Show the webcam frame
    cv2.imshow("ASL Detection", frame)

    # Exit when 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if sentence.strip():
            print('Speaking sentence: ', sentence)
            engine.say(sentence)
            engine.runAndWait()
    elif key == ord('c'):
        sentence = ""
        print('Sentence cleared!')


# --------------------------
# Cleanup
# --------------------------
cap.release()
cv2.destroyAllWindows()
