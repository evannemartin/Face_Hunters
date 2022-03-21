import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # plotting routines
from keras.models import Model       # Model type to be used
from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.utils import np_utils                         # NumPy related tools
import keras
import tensorflow as tf

from sklearn.datasets import fetch_olivetti_faces # Olivetti faces dataset
dataset = fetch_olivetti_faces()
X = dataset["data"]

plt.rcParams['figure.figsize'] = (9,9) # Make the figures a bit bigger
for i in range(25):
    plt.gray()
    plt.subplot(5,5,i+1)
    image2D = X[i,:]
    image2D = image2D.reshape(64,64)
    plt.imshow(image2D)
