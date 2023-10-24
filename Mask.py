import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.animation import FuncAnimation
import imageio
import io
import tensorflow as tf
import pickle

# Load the image
image_path = "British_Shorthair_123.jpg"
image_string = tf.io.read_file(image_path)
image = tf.image.decode_jpeg(image_string, channels=3)


def save_frames(frames, filename="frames.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(frames, f)

def load_frames(filename="frames.pkl"):
    with open(filename, 'rb') as f:
        frames = pickle.load(f)
    return frames

def preprocess_image_only(image, image_size):
    """Apply preprocessing steps to images and resize.
    
    Resize the image to a specified size and preprocess it.
    """
    image = tf.image.resize(image, [image_size, image_size])
    image = mobilenet_preprocess_image(image)
    return image

def mobilenet_preprocess_image(image):
    """Apply preprocessing that is suitable for MobileNetV3.
    
    Simply scales to ranges [-1, 1]
    
    
    you should use this preprocessing for both your model and the mobilenet model
    """
    image = (image - 127.5) / 255.0
    return image


def frame_return(): 
    # Preprocess the image
    preprocessed_image = preprocess_image_only(image, image_size=128)

    # 1. Add a batch dimension
    test_image_batch = tf.expand_dims(preprocessed_image, axis=0)

    model_path = f"/Users/olivervu25/tf-env/mxb362_models/model_at_epoch_30.h5"
    model = tf.keras.models.load_model(model_path)

    # 2. Predict using the model
    outputs = model.predict(test_image_batch)

    # 3. Extract the predicted mask from the outputs
    # Assuming the mask is the second output of your model
    predicted_mask = outputs[1]

    # To remove the batch dimension and get the mask for the single image:
    predicted_mask_single = tf.squeeze(predicted_mask, axis=0)

    # Convert the predicted mask into a binary mask
    predicted_mask_np = predicted_mask_single.numpy()

    # Binary mask with threshold > 0.5
    binary_mask_threshold = (predicted_mask_np > 0.5).astype(np.uint8)

    # Resize the image to 128x128
    resized_image = tf.image.resize(image, [512, 512])
    resized_image = resized_image.numpy().astype(int)

    # Setting up the figure for animation
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Segmentation Performance: Original Image vs Predicted Mask", fontsize=15, weight='bold', y=0.98)

    # Display the image (this remains static throughout the animation)
    ax[0].imshow(resized_image)
    ax[0].set_title('Original Image', fontsize=12)
    ax[0].axis('off')


    # Initial mask display (this will get updated during the animation)
    ax[1].imshow(np.zeros_like(binary_mask_threshold), cmap='viridis')
    ax[1].set_title('Predicted Mask', fontsize=12, weight='bold')
    ax[1].axis('off')

    #

    def update(epoch):
        model_path = f"/Users/olivervu25/tf-env/mxb362_models/model_at_epoch_{epoch}.h5"
        model = tf.keras.models.load_model(model_path)
        outputs = model.predict(test_image_batch)
        predicted_mask = outputs[1]
        predicted_mask_single = tf.squeeze(predicted_mask, axis=0)
        predicted_mask_np = predicted_mask_single.numpy()
        binary_mask_threshold = (predicted_mask_np > 0.5).astype(np.uint8)
        
        # Update the mask display to the newly predicted mask
        ax[1].imshow(binary_mask_threshold, cmap='viridis')
        ax[1].set_title(f"Predicted Mask (Epoch: {epoch})",fontsize=12)

    # Create the animation
    ani = FuncAnimation(fig, update, frames=np.arange(1, 31), blit=False, repeat=False)  # 'repeat' set to False to run the animation only once

    #plt.show()

    frames = []
    # Loop through each epoch and save the frame
    for epoch in range(1, 31):
        update(epoch)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()


    save_frames(frames, "frames2.pkl")  # Save frames to frames.pkl

    return frames 