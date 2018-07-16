import numpy as np
import scipy
import matplotlib.pyplot as plt
from keras.datasets import mnist
from util import func_confusion_matrix



# load (downloaded if needed) the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# transform each image from 28 by28 to a 784 pixel vector
pixel_count = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], pixel_count).astype('float32')
x_test = x_test.reshape(x_test.shape[0], pixel_count).astype('float32')

# normalize inputs from gray scale of 0-255 to values between 0-1
x_train = x_train / 255
x_test = x_test / 255

# Please write your own codes in responses to the homework assignment 5


