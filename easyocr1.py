import easyocr

# Create a reader object for English
reader = easyocr.Reader(['bg'])

# Use the reader to read text from an image
result = reader.readtext('bolnichen_filled.jpg')

# Print the result
for (bbox, text, prob) in result:
    print(text)