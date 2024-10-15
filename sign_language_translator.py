import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from gtts import gTTS
import pygame
import time
import os

# Print debugging information
print("Current working directory:", os.getcwd())
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("OpenCV version:", cv2.__version__)
print("Streamlit version:", st.__version__)

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
if 'word' not in st.session_state:
    st.session_state.word = []

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
            st.session_state.word.append(predicted_letter)
            st.write(f"Predicted letter: {predicted_letter}")
            st.write(f"Current word: {''.join(st.session_state.word)}")
        else:
            st.write("Unable to predict letter. Please try again.")

# Button to add space
if st.button('Add Space'):
    st.session_state.word.append(' ')
    st.write(f"Current word: {''.join(st.session_state.word)}")

# Button to backspace
if st.button('Backspace'):
    if st.session_state.word:
        st.session_state.word.pop()
        st.write(f"Current word: {''.join(st.session_state.word)}")

# Button to clear the word
if st.button('Clear Word'):
    st.session_state.word = []
    st.write("Word cleared")

# Button to speak the word
if st.button('Speak Word'):
    word = ''.join(st.session_state.word)
    if word:
        tts = gTTS(text=word, lang='en')
        tts.save("output.mp3")
        pygame.init()
        pygame.mixer.music.load("output.mp3")
        pygame.mixer.music.play()
        st.write(f"Speaking: {word}")
    else:
        st.write("No word to speak")

# Display current word
st.write(f"Current word: {''.join(st.session_state.word)}")

print("Streamlit app definition completed")