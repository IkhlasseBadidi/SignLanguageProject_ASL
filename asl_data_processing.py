import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the MobileNet model pre-trained on ImageNet, exclude the top dense layers
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of MobileNet
for layer in base_model.layers:
    layer.trainable = False

# Add new custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(29, activation='softmax')(x)  # 29 classes (26 letters + space, nothing, del)

# Define the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Path to training directory
train_data_dir = '/Users/ikhlasse/Desktop/SignLanguageProject/archive/asl_alphabet_train/asl_alphabet_train'

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use 20% of training data for validation
)

# Load and preprocess training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Load and preprocess validation data
validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the trained model
model.save('sign_language_mobilenet.h5')

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Custom function to load test data
def load_test_data(test_dir, target_size=(224, 224)):
    images = []
    labels = []
    class_names = []
    
    for filename in os.listdir(test_dir):
        if filename.endswith("_test.jpg"):
            # Load and preprocess the image
            img_path = os.path.join(test_dir, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
            images.append(img_array)
            
            # Extract class name from filename
            class_name = filename.split('_')[0]
            if class_name not in class_names:
                class_names.append(class_name)
            labels.append(class_names.index(class_name))
    
    return np.array(images), np.array(labels), class_names

# Load test data
test_dir = '/Users/ikhlasse/Desktop/SignLanguageProject/archive/asl_alphabet_test/asl_alphabet_test'
X_test, y_test, class_names = load_test_data(test_dir)

print(f"Loaded {len(X_test)} test images")
print(f"Number of classes: {len(class_names)}")
print(f"Classes: {class_names}")

# Convert labels to categorical
y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes=len(class_names))

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
print(f"Test accuracy: {test_accuracy:.4f}")