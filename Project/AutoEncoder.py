import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import keras
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from multiprocessing import Pool
import os
from PIL import Image
from matplotlib import image

from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose

# PART 1
# In this code, we need to work with numpys. To do so, we need to convert our database
# To an numpy array.
# UPLOAD THE PATH OF THE DATABASE:
data_path="../database/img_align_celeba/img_align_celeba"
listing = os.listdir(data_path)
print(listing) #returns a list of all the files of the path
listarray = [] # creating the array list that will contain the information of our images

def cut_list(list, length):
    listing_parts = []
    intervalle_0 = 0
    intervalle_1 = length
    while intervalle_0 <=(len(list)):
        listing_parts.append(list[intervalle_0:intervalle_1])
        intervalle_0 = intervalle_1
        intervalle_1 = intervalle_1 + length
    return listing_parts

# we choose first to work with only 500 images.
listing_parts=cut_list(listing,500)

#Once we have uploaded all our images, we resize our images and fit them in numpy array
from skimage.transform import resize
for file in listing_parts[0]:
        if file == file + '.DS_Store':
            continue
        chemin= "../database/img_align_celeba/img_align_celeba/" + file
        im = image.imread(chemin)
        resized_img = resize(im,(128,128))
        listarray.append(resized_img)
nparray = np.array(listarray)


# PART 2:  THE ENCODER
#we construct our encoder :
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split (nparray, test_size=0.2, random_state=0)

input_layer = Input(shape=(128, 128, 3), name="INPUT")
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

code_layer = MaxPooling2D((2, 2), name="CODE")(x)

x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(code_layer)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
output_layer = Conv2D(3, (3, 3), padding='same', name="OUTPUT")(x)

AE = Model(input_layer, output_layer)
AE.compile(optimizer='adam', loss='mse')


# We train the encoder
AE.fit(X_train, X_train, epochs=2, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

AE.save("../encodeur.h5")
# We need now to obtain the encoded vector that will be used for the genetic algorithms part:
def auto_encoder():
    get_encoded_X = Model(inputs=AE.input, outputs=AE.get_layer("CODE").output)

    encoded = get_encoded_X.predict(X_test)
    encoded = encoded.reshape((len(X_test), 16*16*8))

    reconstructed = AE.predict(X_test)
    return encoded, reconstructed

encoded, reconstructed=auto_encoder()
np.save("../vecteur.npy", encoded) # THE ENCODED VECTOR IS HERE, A txt file is given. to use it for the genetic algorithm
# you need to reupload it ;)


# PART 3:  THE DECODER

input_layer_decodeur = Input(shape=(16,16,8), name="INPUT")
x_decodeur = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(input_layer_decodeur)
x_decodeur = UpSampling2D((2, 2))(input_layer_decodeur)
x_decodeur = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(input_layer_decodeur)
x_decodeur = UpSampling2D((2, 2))(input_layer_decodeur)
x_decodeur = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(input_layer_decodeur)
x_decodeur = UpSampling2D((2,2))(input_layer_decodeur)
output_layer_decodeur = Conv2D(3, (3, 3), padding='same', name="OUTPUT")(input_layer_decodeur)

D = Model(input_layer_decodeur,output_layer_decodeur)
D.compile(optimizer='adam', loss='mse')

AE.save("../decodeur.h5")


# PART 4: PLOTTING THE PICTURES

def show_face_data(nparray, n=10, title=""):
    plt.figure(figsize=(30, 5))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        plt.imshow(array_to_img(nparray[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title, fontsize = 20)
    plt.show()

show_face_data(X_test, title="original faces")
show_face_data(reconstructed, title="reconstructed faces")
