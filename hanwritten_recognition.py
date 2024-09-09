import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image

# Load the MNIST dataset and split it into training and testing sets for model evaluation
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values to a range between 0 and 1 to improve model training performance
print("Preprocessing the data...")
x_train = x_train / 255.0
x_test = x_test / 255.0

# Print a checkpoint message to track code execution and ensure it reaches this point without issues
print("Data preprocessing completed successfully.")

# Reshape the images to include a channel dimension (necessary for CNN input with grayscale images)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Apply data augmentation to the training data (rotation, zoom, and shifts) to improve generalization
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(x_train)

# Print a message to confirm successful execution of the above steps
print("Data augmentation setup completed.")

# Define the CNN architecture with ReLU activation and L2 regularization
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1),
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10),
])

# Compile the model with appropriate optimizer and loss function
model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model on training set with data augmentation
history = model.fit(datagen.flow(x_train, y_train, batch_size=64),
                    epochs=5,
                    validation_data=(x_test, y_test))

print("Model training completed successfully.")

# Function to recognize a handwritten digit from an image file
def recognize_digit(image_path):
    # Load the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to match the input shape expected by the model (28x28 pixels)
    img = cv2.resize(img, (28, 28))

    # Normalize the pixel values to the range [0, 1] for better prediction accuracy
    img = img / 255.0

    # Expand dimensions to match the model input (batch_size, height, width, channels)
    img = np.expand_dims(img, axis=0)

    # Use the trained model to predict the digit in the image
    prediction = model.predict(img)

    # Get the digit with the highest predicted probability
    digit = np.argmax(prediction)

    # Print the recognized digit
    print(f"Recognized digit: {digit}")

    return digit

# Indicate that the function has successfully recognized the digit
print("Digit recognition process completed successfully.")


# Path to the folder where user-placed images are stored
user_images_folder = "user_images/"

# List image files in the user_images_folder
image_files = [f for f in os.listdir(user_images_folder) if os.path.isfile(os.path.join(user_images_folder, f))]

# Check if there are any images in the folder
if not image_files:
    print("No images found in the folder. Please place an image in the user_images folder.")
else:
    # Loop through all image files and recognize digits
    for image_to_recognize in image_files:
        image_path = os.path.join(user_images_folder, image_to_recognize)

        # Recognize the digit from the chosen image
        recognized_digit = recognize_digit(image_path)

        # Load and display the image with the recognized digit
        img = Image.open(image_path)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"Recognized digit: {recognized_digit}")
        plt.show()

        # Inform the user about the recognized digit
        print(f"The recognized digit from the image '{image_to_recognize}' is: {recognized_digit}")


# Plot performance metrics
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

print("Application terminated.")

# Evaluate the trained model on the entire test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
