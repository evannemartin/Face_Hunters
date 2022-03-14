 import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # plotting routines
from keras.models import Model       # Model type to be used
from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.utils import np_utils                         # NumPy related tools
import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from glob import glob
from keras.preprocessing.image import ImageDataGenerator


# DATASET :
DATA_FOLDER = './database/img_align_celeba'
filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))
NUM_IMAGES = len(filenames)
print("Total number of images : " + str(NUM_IMAGES))

INPUT_DIM = (128,128,3) # Image dimension
BATCH_SIZE = 512
Z_DIM = 200 # Dimension of the latent vector (z)

X = ImageDataGenerator(rescale=1./255).flow_from_directory(DATA_FOLDER,
                                                                   target_size = INPUT_DIM[:2],
                                                                   batch_size = BATCH_SIZE,
                                                                   shuffle = True,
                                                                   class_mode = 'input',
                                                                   subset = 'training'
                                                                   )


#split the database into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

#building the encoder and decoder

original_dim = X.shape[1]
hidden_encoding_dim = 512
encoding_dim = 64
hidden_decoding_dim = 512
dropout_level = 0.1


# Definition of the model that will contain a encoder and a decoder:
# "encoded" is the encoded representation of the input

def model(original_dim, hidden_encoding_dim, encoding_dim,
           dropout_level, hidden_decoding_dim):
           """

           """
    input_img = keras.Input(shape=(original_dim,))
    hidden_encoded = Dense(hidden_encoding_dim, activation='relu')(input_img)
    dropout_hidden_encoded = Dropout(dropout_level)(hidden_encoded)
    encoded = Dense(encoding_dim, activation='relu')(dropout_hidden_encoded)
    dropout_encoded = Dropout(dropout_level)(encoded)

    # "decoded" is the reconstruction of the input
    hidden_decoded = Dense(hidden_decoding_dim, activation='relu')(dropout_encoded)
    dropout_hidden_decoded = Dropout(dropout_level)(hidden_decoded)
    decoded = Dense(original_dim, activation='sigmoid')(dropout_hidden_decoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)
    # This model maps an input to is encoded representation
    encoder = keras.Model(input_img, encoded)
    # This model maps an imput with the same dim as the encoded to the reconstruction
    input_encoded_img = keras.Input(shape=(encoding_dim,))
    hidden_decoder_layer = autoencoder.layers[-3]
    hidden_dropout_decoded_layer = autoencoder.layers[-2]
    decoder_layer = autoencoder.layers[-1]
    decoder = keras.Model(input_encoded_img, decoder_layer(hidden_dropout_decoded_layer(hidden_decoder_layer(input_encoded_img))))

    return encoder, decoder, autoencoder


# compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# Definition of the method plot

encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)
# Use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # How many faces we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(64, 64))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
