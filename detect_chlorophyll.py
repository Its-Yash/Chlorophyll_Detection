import time
import cv2
import numpy as np
import os

# Folder containing images
image_folder = "sample_images"  # Replace with your actual folder path

# Define progress bar parameters
progress_bar_length = 20
progress_bar_char = '='
progress_bar_fill = '.'


def update_progress_bar(img, current_step, total_steps, message):
    """Updates the progress bar and displays it on the window."""
    progress = f"{current_step}/{total_steps}{progress_bar_char * current_step}{progress_bar_fill * (total_steps - current_step)}"
    cv2.putText(img, f"{message} - {progress}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Chlorophyll Detection Processing", img)


def detect_chlorophyll(image_path):
    """Robustly detects chlorophyll in an image.

    Args:
        image_path (str): Path to the image file.
    """

    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")

        # Create a window for progress display (close previous if open)
        cv2.destroyWindow("Chlorophyll Detection Processing")
        cv2.namedWindow("Chlorophyll Detection Processing", cv2.WINDOW_AUTOSIZE)

        # Start time for progress tracking
        start_time = time.time()

        # Convert to HSV
        update_progress_bar(image, 1, 5, "Converting to HSV")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define chlorophyll range (adjust as needed)
        lower_chlorophyll = np.array([35, 50, 40])
        upper_chlorophyll = np.array([90, 255, 255])

        # Noise reduction
        update_progress_bar(image, 2, 5, "Applying noise reduction")
        blur = cv2.GaussianBlur(hsv, (5, 5), 0)  # Adjust kernel size as needed

        # Create mask
        update_progress_bar(image, 3, 5, "Creating mask")
        mask = cv2.inRange(blur, lower_chlorophyll, upper_chlorophyll)

        # Morphological operations (optional)
        kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small holes

        # Apply mask
        update_progress_bar(image, 4, 5, "Applying mask")
        result = cv2.bitwise_and(image, image, mask=mask)

        # Quantification
        update_progress_bar(image, 5, 5, "Calculating chlorophyll percentage")
        white_pixels = np.sum(mask == 255)
        total_pixels = image.shape[0] * image.shape[1]
        chlorophyll_percentage = (white_pixels / total_pixels) * 100

        # Display results
        cv2.imshow('Original Image', image)
        cv2.imshow('Chlorophyll Detection', result)

        # Wait for 2 seconds
        cv2.waitKey(2000)

        # Close windows
        cv2.destroyAllWindows()

        # Print the result with % symbol
        print(f"Image: {image_path}")
        print(f"Chlorophyll percentage:", f"{chlorophyll_percentage:.2f}%")
        print("---")  # For better separation

    except Exception as e:
        print(f"Error processing {image_path}: {e}")


# Loop through all JPEG images in the folder
for image_filename in os.listdir(image_folder):
    # Check if it's a JPEG image
    if image_filename.endswith(".jpeg"):
        image_path = os.path.join(image_folder, image_filename)
        detect_chlorophyll(image_path)