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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Reshape
from keras import layers

# PART 1
# In this code, we need to work with numpys. To do so, we need to convert our database
# To an numpy array.
# UPLOAD THE PATH OF THE DATABASE:
data_path="../database/img_align_celeba/img_align_celeba"
listing = os.listdir(data_path)
#print(listing) #returns a list of all the files of the path
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
print(np.shape(resized_img))
nparray = np.array(listarray)

#######################################################################################################

# PART 2:  THE ENCODER
#we construct our encoder :
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(nparray, test_size=0.2, random_state=0)


input_img = keras.Input(shape=(128, 128, 3))
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x=  layers.MaxPooling2D((2, 2), padding='same')(x)
x=  layers.Flatten()(x)
encoded=  layers.Dense(100, activation='relu', name="CODE")(x)

########################################################################################################

# PART 3:  THE DECODER
x=layers.Dense(512,activation='relu')(encoded)
x=layers.Reshape((8,8,8))(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# We train the encoder
<<<<<<< HEAD
AE.fit(X_train, X_train, epochs=2, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

AE.save("../encodeur.h5")
# We need now to obtain the encoded vector that will be used for the genetic algorithms part:
def auto_encoder():
    get_encoded_X = Model(inputs=AE.input, outputs=AE.get_layer("CODE").output)
=======
autoencoder.fit(X_train, X_train, epochs=2, batch_size=32, shuffle=True, validation_data=(X_test, X_test))
>>>>>>> 6c5eb9a070e0101c57b9e8d7f1e86c91d56dff1b

# We create the decoder model
Decodeur = Model(encoded, decoded)
Decodeur.compile(optimizer='adam', loss='mse')
Decodeur.save("./decodeur.h5")

<<<<<<< HEAD
    reconstructed = AE.predict(X_test)
    return encoded, reconstructed

encoded, reconstructed=auto_encoder()
np.save("../vecteur.npy", encoded) # THE ENCODED VECTOR IS HERE, A txt file is given. to use it for the genetic algorithm
# you need to reupload it ;)
=======
#######################################################################################################
>>>>>>> 6c5eb9a070e0101c57b9e8d7f1e86c91d56dff1b


#PART 4: THE VECTOR
# We need now to obtain the encoded vector that will be used for the genetic algorithms part:

get_encoded_X = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("CODE").output)
encoded = get_encoded_X.predict(X_test)
print(len(X_test))
print(len(encoded[0]))
#encoded = encoded.reshape(100,100)
reconstructed = autoencoder.predict(X_test)

np.save("./vecteur.npy", encoded) # THE ENCODED VECTOR IS HERE, A npy file is given. to use it for the genetic algorithm
# you need to reupload it ;)

<<<<<<< HEAD
AE.save("../decodeur.h5")
=======
>>>>>>> 6c5eb9a070e0101c57b9e8d7f1e86c91d56dff1b

#######################################################################################################

# PART 5: PLOTTING THE PICTURES

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
