import pytesseract
from PIL import Image
import pathlib

img =Image.open ("1.png")
text = pytesseract.image_to_string(img, config="")
print (text)