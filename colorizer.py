from nn_model import model, nn_node
import numpy as np
from PIL import Image
import os

class colorizer:
    """
    """

    def __init__(self):
        self.model = model()
        self.image = colorizer.load_image("./test.png")
        self.rgb_data = np.asarray(self.image, dtype="int32")
        self.gray = colorizer.rgb_to_grayscale(self.rgb_data)
        self.padded_gray = np.pad(self.gray, (1, 1), "constant", constant_values = 0)


    def train(self):
        # get training data and grayscale image from utils
        for _ in range(10):
            x = random.randint()
            y = random.randint()
            sample_gray = self.padded_gray[x:x+3, y:y+3]
            sample_rgb = self.rgb_data[x,y]

            pred = model.forward(sample_gray)
            loss = (np.sum(pred-sample_rgb)**2)
            model.update_output_layer(loss)

    @staticmethod
    def load_image(path):
        im = Image.open(path)
        im.load()
        return im

    @staticmethod
    def rgb_to_grayscale(data):
        r = data[:,:,0]
        g = data[:,:,1]
        b = data[:,:,2]
        grayscale = [0.3, 0.55, 0.15]
        gray = data[:, :, 0:3].dot(grayscale)
        gray = np.floor(gray)
        # gray = .3 * r + 0.55 * g + 0.15 * b
        # Image.fromarray(gray).convert("RGB").show()
        return gray


if __name__ == '__main__':
    MODEL =  colorizer()
