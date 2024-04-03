import cv2
import numpy as np


def detect_yellowness(image):
    """Detects yellow levels in an image.

    Args:
        image: The input RGB image (NumPy array).

    Returns:
        yellowness_ratio: A float representing the percentage of yellow pixels.
        yellow_mask: A binary image where yellow regions are highlighted.
    """

    # Convert to HSV color space for better yellow isolation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define yellow color range in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a mask for the yellow pixels
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Optional: Refine the mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    # Calculate yellowness metrics
    total_pixels = yellow_mask.size
    yellow_pixels = np.sum(yellow_mask == 255)
    yellowness_ratio = (yellow_pixels / total_pixels) * 100

    return yellowness_ratio, yellow_mask


# --- Example Usage ---
# Load an image
image = cv2.imread('path/to/your/image.jpg')

# Detect yellowness
yellowness_ratio, yellow_mask = detect_yellowness(image)

print("Yellowness Ratio:", yellowness_ratio, "%")

# Display the results (optional)
cv2.imshow('Original Image', image)
cv2.imshow('Yellow Mask', yellow_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
