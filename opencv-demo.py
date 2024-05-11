import cv2
from matplotlib import pyplot as plt
import numpy as np

image_file = "test.jpg"

img = cv2.imread(image_file)


# cv2.imshow("test image", img)
# cv2.waitKey(1000)

# function to display images as plot
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width = im_data.shape[:2]

    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis('off')

    ax.imshow(im_data, cmap='gray')

    plt.show()


inverted_image = cv2.bitwise_not(img)
cv2.imwrite("inverted_image.jpg", inverted_image)


# display("inverted_image.jpg")


# ********** binerization
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


gray_image = grayscale(img)

cv2.imwrite("gray_image.jpg", gray_image)

# display("gray_image.jpg")

thresh, im_bw = cv2.threshold(gray_image, 200, 230, cv2.THRESH_BINARY)

cv2.imwrite("bwimage.jpg", im_bw)


# display("bwimage.jpg")

# ********** noise removal
def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)


no_noise = noise_removal(im_bw)
cv2.imwrite("no_noise.jpg", no_noise)

display("no_noise.jpg")

# ************ Dilation and Erosion
