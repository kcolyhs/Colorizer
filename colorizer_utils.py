from PIL import Image
import numpy as np


def load_image(path):
    im = Image.open(path)
    im.load()
    # im.show()
    return im


def import_data(image):
    data = np.asarray(image, dtype="int32")
    gray_image = rgb_to_grayscale(data)
    X = gray_image
    Y = data[:,:,0:3]
    return X ,Y


def rgb_to_grayscale(data):
    r = data[:,:,0]
    g = data[:,:,1]
    b = data[:,:,2]
    grayscale = [0.3, 0.55, 0.15]
    gray = data[:, :, 0:3].dot(grayscale)
    # gray = .3 * r + 0.55 * g + 0.15 * b
    Image.fromarray(gray).convert("RGB").show()
    return gray

def get_sample(data):
    return None

im = load_image("./test.png")
data = np.asarray( im, dtype="int32")
gray = rgb_to_grayscale(data)
