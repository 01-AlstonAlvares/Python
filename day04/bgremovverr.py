import cv2
import numpy as np

def remove_background(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use thresholding to create a binary image
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert the binary image
    binary = cv2.bitwise_not(binary)
    
    # Apply morphological operations to remove noise and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Create a mask containing only the background
    contours, _ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(sure_bg)
    for contour in contours:
        cv2.drawContours(mask, [contour], 0, (255), -1)
    
    # Remove the background
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result

# Example usage
input_image_path = r"day04\logo.png"  # Using raw string to handle backslashes
output_image = remove_background(input_image_path)
cv2.imwrite("output_image.png", output_image)
