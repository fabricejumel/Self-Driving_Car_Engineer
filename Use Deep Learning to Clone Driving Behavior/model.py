import os
import tensorflow as tf
import pandas as pd
import pandas
import numpy as np
import skimage.transform as sktransform
import random
import matplotlib.image as mpimg
import os
import cv2
import matplotlib.pyplot as plot
from keras import optimizers
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from PIL import Image

class DataGenerator(object):
    def __init__(self, data, image_dir, training_mode=True, batch_size=128):
        self.data = data
        self.img_path = image_dir
        self.training_mode = training_mode
        self.angles = ['left', 'center', 'right']
        self.steering_angle = [.25, 0., -.25]
        self.batch_size = batch_size
        self.cnt = 0

    @staticmethod
    def brightness(image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv_image[:, :, 2] = hsv_image[:, :, 2] * .30
        rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
        return rgb_image

    def random_image(self):
        image_choice = np.random.randint(len(self.data))
        camera_choice = np.random.randint(len(self.angles))
        image_filename = self.img_path + self.data[self.angles[camera_choice]].values[image_choice].strip()
        image = mpimg.imread(image_filename)
        steering_angle = self.data.steering.values[image_choice] + self.steering_angle[camera_choice]
        return image, steering_angle

    def add_image(self):
        if self.cnt < self.batch_size:
            self.__x[self.cnt] = self.image
            self.__y[self.cnt] = self.__steering_angle
            self.cnt += 1
            return True
        return False

    def add_bright(self):
        image = np.copy(self.image)
        image = DataGenerator.brightness(image)
        if self.cnt < self.batch_size:
            self.__x[self.cnt] = image
            self.__y[self.cnt] = self.__steering_angle
            self.cnt += 1
            return True
        return False

    def __call__(self):
        while True:
            self.__x = np.zeros((self.batch_size, 160, 320, 3), dtype=np.float32)
            self.__y = np.zeros(self.batch_size, dtype=np.float32)
            self.cnt = 0
            while self.__cnt < self.batch_size:
                self.image, self.__steering_angle = self.random_image()
                if self.training_mode:
                    flip = np.random.randint(2)
                    if flip == 0:
                        self.image = cv2.flip(self.image, 1)
                        self.__steering_angle = -self.__steering_angle

                    if not self.add_image():
                        break

                    if not self.add_bright():
                        break
                else:
                     self.add_image()

            yield(self.__x, self.__y)

def model():
    model1 = Sequential()
    model1.add(Lambda(lambda x: x/255-0.5, input_shape=(160, 320, 3)))
    model1.add(Cropping2D(((80, 20), (0, 20))))
    model1.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model1.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model1.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model1.add(Flatten())
    model1.add(Dense(100))
    model1.add(Activation('relu'))
    model1.add(Dense(50))
    model1.add(Activation('relu'))
    model1.add(Dense(10))
    model1.add(Activation('relu'))    
    model1.add(Dense(1))
    return model1

if __name__ == '__main__':
    path = 'data/'
    data = pd.read_csv(path + 'driving_log.csv')
    data_train, data_valid = model_selection.train_test_split(data, test_size=.2)
    model = model()
    model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')
    history = model.fit_generator(
        DataGenerator(data_train, path)(),
        samples_per_epoch=data_train.shape[0]*2,
        nb_epoch=5,
        validation_data=DataGenerator(data_valid, path, training_mode=False)(),
        nb_val_samples=data_valid.shape[0]
    )
    model.save('model.h5')