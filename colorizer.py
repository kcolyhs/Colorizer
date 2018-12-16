from nn_model import model
import numpy as np
from PIL import Image
from random import randint, seed
import matplotlib.pyplot as plt
import os

seed(1)
class colorizer:
    def __init__(self):
        self.model = model()
        self.import_data("./training/")

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
        grayimage = Image.fromarray(gray).convert("RGB")
        grayimage.save("gray.png")
        return gray


if __name__ == '__main__':
    test = colorizer()

    trials = 1000
    epochs = 1000
    loss_history = []
    epoch_history = []

    for e in range(epochs):
        epoch_avg = 0
        for _ in range(trials):
            i = randint(0,69960-1)
            # print(f"trial#{_} using sample#{i}")
            output, loss = test.model.forward(test.X[i], test.Y[i], training=True)
            loss_history.append(loss)
            epoch_avg += loss/trials
        print(f"Avg loss for epoch#{e}: {epoch_avg}")
        epoch_history.append(epoch_avg)
    # print(test.model.forward(test.X[0], test.Y[0], training=True))

    y_pred = []
    for i in range(test.X.shape[0]):
            output= test.model.forward(test.X[i], test.Y[i], training=False)
            output = output[0]
            output = np.floor(output)
            y_pred.append(output)
            if i % 1000 is 1000:
                print(i)
    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(318,220,3)
    np.clip(y_pred, 0, 255, out=y_pred)
    y_pred = y_pred.astype('uint8')
    result = Image.fromarray(y_pred).convert("RGB")
    result.show()
    result.save("result.png")

    fig = plt.plot(epoch_history[:])
    plt.show()
    print("finished")
