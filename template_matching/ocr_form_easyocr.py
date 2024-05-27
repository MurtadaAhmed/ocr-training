import cv2
import pytesseract
import easyocr


def processing_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def template_matching(image, template_path):
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_location = cv2.minMaxLoc(result)
    return max_location, template.shape


def extract_text(image, match_location):
    extracted_text = {}

    reader = easyocr.Reader(['bg', 'en'])

    for key, (location, shape) in match_location.items():
        x, y = location
        w, h = shape[::-1]
        roi = image[y:y + h, x:x + w]
        cv2.imwrite(f"{key}.jpg", roi)
        text = reader.readtext(roi, detail=0)
        extracted_text[key] = "".join(text[1:]).replace(" ", "")

    return extracted_text


image_path = "bolnichen_filled.jpg"
template_paths = {
    'sick_leave_number': './templates/sick_leave_number.jpg',
    'first_time': './templates/first_time.jpg',
    'continuation': './templates/continuation.jpg',
    'duration_from': './templates/duration_from.jpg',
    'duration_to': './templates/duration_to.jpg',
    'diagnosis': './templates/diagnosis.jpg',
    'reason': './templates/reason.jpg',
    'date': './templates/date.jpg',
}

processed_image = processing_image(image_path)

matched_locations = {}

for key, template_path in template_paths.items():
    location, shape = template_matching(processed_image, template_path)
    matched_locations[key] = (location, shape)


def image_without_borders(image_path):
    img = cv2.imread(image_path)

    _, result = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)



    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))

    detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

    vertical_contours, _ = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for ver_contour in vertical_contours:
        cv2.drawContours(gray, [ver_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    hor_contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in hor_contours:
        cv2.drawContours(gray, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)



    return binary


finally_processed_image = image_without_borders(image_path)

extracted_text = extract_text(finally_processed_image, matched_locations)

print(extracted_text)
