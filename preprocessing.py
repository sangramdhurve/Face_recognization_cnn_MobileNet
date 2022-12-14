from PIL import Image
from io import BytesIO
import numpy as np

# Reading the Image Uploaded by the users
def read_imagefile(file) -> Image.Image:
    Readimage = Image.open(BytesIO(file))
    return Readimage

def Preprocessing(image: Image.Image):
    image = image.resize((224, 224))             # Resizing The Image , Use also (image.resize(image_shape))
    image = np.array(image)                              # to convert input(array) to a float type array
    image = np.expand_dims(image, 0)                        # Use for Expand the shape of an array
    image = image / 255                                     # Normalizing
    return image
