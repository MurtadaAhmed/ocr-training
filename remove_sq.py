import cv2
import pytesseract
import easyocr
from pdf2image import convert_from_path

# pdf = 'bolnichen.pdf'
# image = convert_from_path(pdf)
# image[0].save('bolnichen.jpg', 'JPEG')

img = cv2.imread('bolnichen_filled.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, result = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)

cv2.imwrite("bolnichen_filled1.jpg", result)

image = cv2.imread("bolnichen_filled1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)


horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
# vertical2_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 9))
# horizontal_2_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2))
# horizontal_3_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))


detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
# detect_vertical2 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical2_kernel, iterations=2)
# detect_horizontal2 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_2_kernel, iterations=2)
# detect_horizontal3 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_3_kernel, iterations=2)

vertical_contours, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for ver_contour in vertical_contours:
    cv2.drawContours(image, [ver_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
# detect_vertical2
# vertical_contours2, _ = cv2.findContours(detect_vertical2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Fill the detected contours with white to remove the borders
# for ver_contour in vertical_contours2:
#     cv2.drawContours(image, [ver_contour], -1, (255, 255, 255), thickness=cv2.FILLED)



detected_lines = cv2.add(detect_horizontal, detect_vertical)


hor_contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for contour in hor_contours:
    cv2.drawContours(image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# Detect contours of the lines
# hor2_contours, _ = cv2.findContours(detect_horizontal2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Fill the detected contours with white to remove the borders
# for contour in hor2_contours:
#     cv2.drawContours(image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

# Detect contours of the lines
# hor3_contours, _ = cv2.findContours(detect_horizontal3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Fill the detected contours with white to remove the borders
# for contour in hor3_contours:
#     cv2.drawContours(image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
morph = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


cleaned_image_path = 'bolnichen_filled1.jpg'
cv2.imwrite(cleaned_image_path, image)

image = cv2.imread("bolnichen_filled1.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray, config='--psm 6 -l bul')
print("Tesseract Result:")
print("-----------------")
print(text)
print("*********************************")
print("EasyOCR Result:")
print("---------------")
reader = easyocr.Reader(['bg', 'en'])
text = reader.readtext(gray, detail=0)
for element in text:
    print(element)

