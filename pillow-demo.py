from PIL import Image

im_file = "test.png"

im = Image.open(im_file)
im.save("test222.jpg")
im.rotate(180).show()

print(im.size)