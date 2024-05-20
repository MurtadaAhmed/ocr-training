import cv2
import numpy as np

# Load and preprocess the image
img = cv2.imread('bolnichen.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Use Hough Line Transform to detect lines
lines = cv2.HoughLinesP(binary, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

# Draw the lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

# Save the cleaned image
cleaned_image_path = 'bolnichen_cleaned.png'
cv2.imwrite(cleaned_image_path, img)