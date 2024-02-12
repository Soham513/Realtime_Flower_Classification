import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Load your pre-trained Keras model
model_path = 'C:/Users/Soham/Desktop/My_Flowers_Dataset/my_flower_detection_model.h5'
model = keras.models.load_model(model_path)

# Define the classes for the flowers
CLASSES = ["Daisy", "Dhalia Pinnata", "Hibiscus", "Rose", "Surfinia"]

# Open a video capture object (0 corresponds to the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Resize the frame to match the input size expected by the model
    input_tensor = cv2.resize(frame, (150, 150))
    input_tensor = np.expand_dims(input_tensor, axis=0) / 255.0  # Normalize to [0, 1]

    # Perform inference
    predictions = model.predict(input_tensor)

    # Process the predictions
    class_id = np.argmax(predictions)
    label = CLASSES[class_id]

    # Display the label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with detected flower using Matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Flower Detection: " + label)
    plt.pause(0.01)  # Required for Matplotlib to update the plot
    plt.clf()  # Clear the figure for the next iteration

    # Uncomment the next line if you want to break the loop on 'q' key press
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
