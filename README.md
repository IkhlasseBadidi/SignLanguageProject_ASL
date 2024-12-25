# ASL Alphabet to Voice Translation


A real-time American Sign Language (ASL) alphabet detection and voice translation system, developed as part of the Introduction to Data Science Course at the University of Helsinki.

## Team Members
- Ikhlasse Badidi
- Aya Riany
- Mike Rizk

## Project Overview

This project addresses the communication gap between deaf and hearing individuals by providing a real-time ASL alphabet to voice translation system. The application captures ASL alphabet signs through a webcam, processes them in real-time, and converts them into text and speech output.

### Key Features
- Real-time ASL alphabet sign detection
- Progressive sentence building letter by letter
- Text-to-speech conversion
- User-friendly web interface
- Immediate visual feedback

## Technical Details

### Dataset
- Sourced from [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- 87,000 training images (200x200 pixels)
- 29 classes (A-Z + SPACE, DELETE, NOTHING)
- Data augmentation using TensorFlow's ImageDataGenerator
- 80-20 train-validation split

### Model
- Implemented using MobileNet CNN architecture
- Optimized for mobile devices and low-memory usage
- Designed for efficient real-time processing

### Technologies Used
- TensorFlow: Core model implementation and inference
- NumPy: Image processing and array operations
- OpenCV: Webcam access and image capture
- gtts (Google Text-to-Speech): Text to voice conversion
- Streamlit: Web application interface
- Pygame: Audio playback

## Demo Functionality
Demo Link: https://drive.google.com/file/d/1Xya1TVmeKApgv8pplW8f8GdgrSbauPXS/view?usp=share_link 
\
The application allows users to:
1. Capture ASL alphabet signs through webcam
2. Build sentences letter by letter
3. View real-time letter detection
4. Convert completed sentences to speech
5. Clear progress and start over

## Future Development
- Mobile application development
- Word detection from video streams
- Support for multiple languages
- Speech to Sign Language translation
- Enhanced deployment solutions

## Repository Structure
- `asl_data_processing.py`: Data preprocessing pipeline
- `sign_language_mobilenet.h5`: Trained model file
- `sign_language_translator.py`: Core translation logic
- `asl_alphabet_test/`: Test dataset
- `asl_alphabet_train/`: Training dataset

## Acknowledgments
Special thanks to the Kaggle dataset creators and the Introduction to Data Science course instructors at the University of Helsinki.
