import cv2
import numpy as np

# Reading, displaying, and writing an image
image = cv2.imread('./data/tic_tac_toe.jpg')  # Replace 'input.jpg' with your image file
img_temp = image.copy()
cv2.imshow('Original Image', image)

# Understanding image properties
print(f"Image dimensions: {image.shape}")
print(f"Number of pixels: {image.size}")
print(f"Data type of image: {image.dtype}")

# Basic image transformations (resize and crop)
resized_image = cv2.resize(image, (300, 300))  # Resize to 300x300
cropped_image = image[50:150, 50:150]  # Crop the region from 50,50 to 150,150
cv2.imshow('Resized Image', resized_image)
image = img_temp.copy()  # Reset the image to original

# Color spaces and conversions
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow('Gray Image', gray_image)
image = img_temp.copy()  # Reset the image to original

# Thresholding
_, binary_thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Binary Threshold', binary_thresh)
image = img_temp.copy()  # Reset the image to original

# Image filtering (blurring and sharpening)
blurred_image = cv2.GaussianBlur(image, (7, 7), 0)
sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)
cv2.imshow('Sharpened Image', sharpened_image)
image = img_temp.copy()  # Reset the image to original

# Edge detection
edges = cv2.Canny(image, 100, 200)
cv2.imshow('Edges', edges)
image = img_temp.copy()  # Reset the image to original

# Drawing functions
cv2.line(image, (0, 0), (150, 150), (255, 0, 0), 5)
cv2.rectangle(image, (50, 50), (100, 100), (0, 255, 0), 3)
cv2.circle(image, (120, 120), 30, (0, 0, 255), -1)
cv2.putText(image, 'OpenCV', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imshow('Drawing Functions', image)
image = img_temp.copy()  # Reset the image to original

# Contour detection
contours, _ = cv2.findContours(binary_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contour Image', image)
image = img_temp.copy()  # Reset the image to original
# Display the results of all operations
cv2.waitKey(0)
cv2.destroyAllWindows()
