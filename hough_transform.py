import cv2

# Load image
image = cv2.imread('bolnichen_filled.jpg')

# Convert the image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary threshold
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
for contour in contours:
    # Get rectangle bounding contour
    [x, y, w, h] = cv2.boundingRect(contour)

    # Calculate contour area
    area = cv2.contourArea(contour)

    # Only draw contour if area is above a certain size
    if area > 500:  # Adjust this value based on your specific use case
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Save the image with contours
cv2.imwrite('contours.jpg', image)






