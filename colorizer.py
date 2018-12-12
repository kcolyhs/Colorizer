from nn_model import model, node
from colorizer_utils import load_image, get_sample, rgb_to_grayscale
import numpy as np

from PIL import Image

class colorizer:
    """
    """

    def __init__(self):
        self.model = model()
        self.image = load_image()
        self.data = np.asarray(self.image, dtype="int32")
        self.gray = rgb_to_grayscale(self.data)

    def train(self):
        # get training data and grayscale image from utils
        for _ in range(10):
            sample = get_sample(self.data)
            sample_gray = sample[0]
            sample_rgb = sample[1:4]
            pred = model.forward(sample_gray)
            loss = (np.sum(pred-sample_rgb)**2)
            model.update_output_layer(loss)


if __name__ == '__main__':
    IMAGE = load_image("./test.jpg")
    MODEL =  colorizer()
