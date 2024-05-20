import cv2

# Load and preprocess the image
img = cv2.imread('bolnichen.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Define kernels for detecting horizontal and vertical lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

# Use morphological operations to detect horizontal and vertical lines
detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

# Combine horizontal and vertical lines
detected_lines = cv2.add(detect_horizontal, detect_vertical)

# Detect contours of the lines
contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Fill the detected contours with white to remove the borders
for contour in contours:
    cv2.drawContours(img, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# Save the cleaned image
cleaned_image_path = 'bolnichen_cleaned.png'
cv2.imwrite(cleaned_image_path, img)