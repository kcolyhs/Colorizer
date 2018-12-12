from nn_model import model
import numpy as np
from PIL import Image
import os


class colorizer:
    def __init__(self):
        self.model = model()
        self.import_data("./training/")

    def train(self):
        # get training data and grayscale image from utils
        for _ in range(10):
            #TODO implement
            # sample_gray = None
            # sample_rgb = self.rgb_data[x,y]

            # pred = model.forward(sample_gray)
            # loss = (np.sum(pred-sample_rgb)**2)
            # model.update_output_layer(loss)
            pass

    def import_data(self, path):
        print("Importing training files:")
        X = np.empty((0, 9), int)
        Y = np.empty((0, 3), int)
        for filename in os.listdir(path):
            print(f"Importing {filename}...")
            im = Image.open(path+filename)
            im.load()
            rgb_data = np.asarray(im, dtype="int32")
            gray = colorizer.rgb_to_grayscale(rgb_data)
            padded_gray = np.pad(gray, (1, 1), "constant", constant_values=0)
            i_size, j_size = np.shape(gray)
            im_X = np.empty((gray.size, 9), int)
            im_Y = np.empty((gray.size, 3), int)
            index = 0
            for i in range(i_size):
                for j in range(j_size):
                    im_X[index] = (padded_gray[i:i+3, j:j+3].flatten())
                    im_Y[index] = (rgb_data[i, j, 0:3])
                    index += 1
            X = np.concatenate((X, im_X))
            Y = np.concatenate((Y, im_Y))
            print(f"Imported {filename}: {gray.size} samples imported")
        self.X = np.array(X)
        self.Y = np.array(Y)

    @staticmethod
    def rgb_to_grayscale(data):
        grayscale = [0.3, 0.55, 0.15]
        gray = data[:, :, 0:3].dot(grayscale)
        gray = np.floor(gray)
        # gray = .3 * r + 0.55 * g + 0.15 * b
        # Image.fromarray(gray).convert("RGB").show()
        return gray


if __name__ == '__main__':
    test = colorizer()
    print(test.model.forward(test.X[0], test.Y[0]), training=True)
    print("finished")
