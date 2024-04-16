import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from skimage.transform import resize
import numpy as np
from keras.models import load_model 
from keras.optimizers import Adam
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the best model checkpoints
best_model2 = load_model("checkpoints/mobilenet/checkpoint.keras")

# Compile the model with additional metrics
best_model2.compile(
    optimizer=Adam(),  # Optimizer for training
    loss='categorical_crossentropy',  # Loss function for optimization
    metrics=['accuracy', 'precision', 'recall', 'f1_score']  # Metrics to monitor during training
)

def sign_predict(img):
    """
    Function to predict the sign from the input image.
    Args:
        img (numpy.ndarray): Input image containing the sign.
    Returns:
        int: Predicted class label for the sign.
    """
    
    # Set the path to your data directory
    test_dir = "live_test/"

    # Data augmentation and preprocessing
    test_data_generator = ImageDataGenerator(
        rescale=1./255,
        fill_mode='nearest',
    )

    # Provide the path to your dataset
    test_generator = test_data_generator.flow_from_directory(
        test_dir,
        target_size=(64, 64),
        batch_size=1,
        class_mode='categorical',
        color_mode='rgb',  # Set color mode to RGB
    )
    
    predictions = best_model2.predict(test_generator.__getitem__(0)[0])
    predicted_labels_batch = np.argmax(predictions, axis=1)
 
    # print(predicted_labels_batch, predictions[0].predict)
    return predicted_labels_batch[0]

def save_uploaded_file(img):
    """
    Function to save the uploaded image to a directory.
    Args:
        img (numpy.ndarray): Input image to be saved.
    """
    
    # Convert the numpy array to an image
    image = Image.fromarray(img.astype(np.uint8))
    # Save the image
    image.save("live_test/0/saved_image.jpg") 

def crop_hands(image):
    """
    Function to detect and crop hands from the input image.
    Args:
        image (numpy.ndarray): Input image containing hands.
    Returns:
        numpy.ndarray: Cropped image containing only the hand region.
    """
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:
            x_vals = [landmark.x for landmark in hand_landmarks.landmark]
            y_vals = [landmark.y for landmark in hand_landmarks.landmark]
            
            min_x, max_x = min(x_vals), max(x_vals)
            min_y, max_y = min(y_vals), max(y_vals)
            
            # Add some padding to the bounding box
            padding = 100
            min_x, max_x = int(min_x * image.shape[1]) - 2*padding, int(max_x * image.shape[1]) + 2*padding
            min_y, max_y = int(min_y * image.shape[0]) - int(0.75*padding), int(max_y * image.shape[0]) + int(0.75*padding)
            
            # Crop the hand region
            cropped_hand = image[min_y:max_y, min_x:max_x]
            return cropped_hand

    return image

# Streamlit app title
st.title("ASL-Numbers Recognition")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

predicted_class = None

col1, col2 = st.columns(2)

with col1:
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, use_column_width=True)
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        cropped_hand = img
        print(type(img))
        
        # Crop hands from the image
        cropped_hand = crop_hands(img)
        save_uploaded_file(cropped_hand)
            
        
        with col2:
            
            st.image(cropped_hand, channels="BGR", use_column_width=True)
            
            if cropped_hand is not None:
                
                predicted_class = sign_predict(cropped_hand)
                st.write("Predicted class: {}".format(predicted_class))
