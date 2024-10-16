import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from gtts import gTTS
import pygame
import time
import os

try:
    model = tf.keras.models.load_model('sign_language_mobilenet.h5', compile=False)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to classify the image
def classify_image(image):
    try:
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions)
        # Ensure the predicted class is within the valid range (0-25 for A-Z)
        if 0 <= predicted_class <= 25:
            return chr(ord("A") + predicted_class)
        else:
            print(f"Unexpected prediction: {predicted_class}")
            return None
    except Exception as e:
        print(f"Error in classify_image: {e}")
        st.error(f"Error in image classification: {e}")
        return None

# Streamlit app
st.title('Sign Language Alphabet Translator')

# Initialize session state
if 'prefix' not in st.session_state:
    st.session_state.prefix = "My name is "
if 'name' not in st.session_state:
    st.session_state.name = ""

# Function to capture image
def capture_image():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open camera")
            return None
        # Countdown
        for i in range(3, 0, -1):
            st.write(f"Get ready... {i}")
            time.sleep(1)
        st.write("Show your sign now!")
        # Capture frame
        ret, frame = cap.read()
        cap.release()
        if not ret:
            st.error("Can't receive frame (stream end?). Exiting ...")
            return None
        return frame
    except Exception as e:
        print(f"Error in capture_image: {e}")
        st.error(f"Error capturing image: {e}")
        return None

# Button to capture image and predict
if st.button('Capture Sign'):
    image = capture_image()
    if image is not None:
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Captured Sign")
        predicted_letter = classify_image(image)
        if predicted_letter:
            st.write(f"Predicted letter: {predicted_letter}")
            st.session_state.name += predicted_letter
            st.write(f"Current progress: {st.session_state.prefix}{st.session_state.name}")
        else:
            st.write("Unable to predict letter. Please try again.")

# Button to clear progress
if st.button('Clear Progress'):
    st.session_state.name = ""
    st.write("Progress cleared")

# Button to read the full sentence
if st.button('Read Full Sentence'):
    full_sentence = f"{st.session_state.prefix}{st.session_state.name}"
    tts = gTTS(text=full_sentence, lang='en')
    tts.save("output.mp3")
    pygame.init()
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()
    st.write(f"Reading: {full_sentence}")

# Display current progress
st.write(f"Current progress: {st.session_state.prefix}{st.session_state.name}")

print("Streamlit app definition completed")