import cv2
import json
import xml.etree.ElementTree as ET
import easyocr

# Function to read annotations from a PASCAL VOC XML file
def read_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []
    for member in root.findall('object'):
        bndbox = member.find('bndbox')
        annotation = {
            'label': member.find('name').text,
            'xmin': int(bndbox.find('xmin').text),
            'ymin': int(bndbox.find('ymin').text),
            'xmax': int(bndbox.find('xmax').text),
            'ymax': int(bndbox.find('ymax').text)
        }
        annotations.append(annotation)
    return annotations

# Function to perform OCR on the specified region of an image using EasyOCR
def ocr_on_region(reader, image, region):
    cropped_image = image[region['ymin']:region['ymax'], region['xmin']:region['xmax']]
    # Convert image to grayscale as EasyOCR works better with single channel images
    gray_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    text = reader.readtext(gray_cropped_image, detail=0)
    return ' '.join(text).strip().replace("/", "").replace("|", "").replace(" ", "")

# Main function to process images and annotations
def process_images_and_annotations(image_files, annotation_files):
    reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader with necessary languages
    data = []
    for image_file, annotation_file in zip(image_files, annotation_files):
        image = cv2.imread(image_file)
        annotations = read_annotations(annotation_file)
        extracted_data = {}
        for annotation in annotations:
            text = ocr_on_region(reader, image, annotation)
            extracted_data[annotation['label']] = text
        data.append(extracted_data)
    return data

# Example usage
image_files = ['./bolnichen_filled.jpg']
annotation_files = ['./bolnichen.xml']

extracted_data = process_images_and_annotations(image_files, annotation_files)

# Convert to JSON and save to file
json_data = json.dumps(extracted_data, indent=4)
with open('extracted_data.json', 'w') as json_file:
    json_file.write(json_data)

print(json_data)
