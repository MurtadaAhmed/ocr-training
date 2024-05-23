# installing albumentations with its dependencies (opencv included)
# pip install -U albumentations

# import necessary libraries
import albumentations as A
import cv2

# define augmentation pipeline
transform = A.Compose([
    # A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5), # apply horizontal flip to 50% of images (random)
    A.RandomBrightnessContrast(p=0.2), # apply random brightness and contrast
    A.Rotate(limit=4, p=0.5), # rotate the image by 4 degrees
    A.GaussianBlur(blur_limit=(1,3), p=0.05), # apply gaussian blur with a kernel size between 1 and 3
])

# load the image and convert it to gray
image = cv2.imread("bolnichen.jpg") # select the image here
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# pass the image to the augmentation pipline
transformed = transform(image=gray) # return a dictionary with the key 'image'

# get the transformed image
transformed_image = transformed['image']

# save the transformed image
cv2.imwrite("bolnichen_transformed.jpg", transformed_image)

