import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # plotting routines
from keras.models import Model       # Model type to be used
from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.utils import np_utils                         # NumPy related tools
import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import PIL
from glob import glob
from keras.preprocessing.image import ImageDataGenerator


# Definition of the model that will contain a encoder and a decoder:

def model(original_dim, hidden_encoding_dim, encoding_dim, dropout_level, hidden_decoding_dim):
    """
         This method contains an encoder and a decoder (neuronal model)
         The encoder program takes an image as a parameter and returns a low dimensional vector
         The decoder program takes the vector as a parameter and learns to reconstruct an image from this one.

         args :
          original_dim (int)
          hidden_encoding_dim (int)
          encoding_dim (int)
          dropout_level (int)
          hidden_decoding_dim (int)

        returns :
          (keras.engine.functional.Functional) : encoder
          (keras.engine.functional.Functional) : decoder
          (keras.engine.functional.Functional) : autoencoder
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


# Definition of the method plot
def plot_images(n,decoded):
    """
    This method plots the n images before and after the reconstruction of the autoencoder
    args :
        decoded (array)
        n (int) : number of faces we will display
    returns :
        None

    """
    encoded_imgs = encoder.predict(X_test)
    decoded_imgs = decoder.predict(encoded_imgs)
    import matplotlib.pyplot as plt

    n = 10  # How many faces we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_test[i].reshape(128, 128))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(128, 128))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    return None

# LOAD the DATASET (only 504 first pictures) :
DATA_FOLDER="./database/img_align_celeba"
"""X=[]
for i in range(1,10): #501 so that it doesn't crash for now
    img = PIL.Image.open(DATA_FOLDER+"00000"+str(i)+".jpg") # This returns an image object
    img = np.asarray(img) # convert it to ndarray
    print(type(img))
    X.append(X)
"""
filenames = np.array(glob(os.path.join(DATA_FOLDER, '*/*.jpg')))
NUM_IMAGES = len(filenames)
#print("Total number of images : " + str(NUM_IMAGES))

INPUT_DIM = (128,128,3) # Image dimension
BATCH_SIZE = 500
data_flow = ImageDataGenerator(rescale=1./255).flow_from_directory(DATA_FOLDER,
                                                                   target_size = INPUT_DIM[:2],
                                                                   batch_size = BATCH_SIZE,
                                                                   shuffle = True,
                                                                   class_mode = 'input',
                                                                   subset = 'training'

# try to upload the images in an other way                                                                    )
#print(data_flow[1])
#print(data_flow.__len__())
"""X=np.empty(504,dtype=object)
for i in range(len(data_flow)):
    X[i]=data_flow[i]"""


#print(X[1])
#X=np.asarray(X)
#print(np.shape(X[0]))

# TEST THE DIFFERENT FUNCTIONS

# first, we need to split the database into training and testing
#from sklearn.model_selection import train_test_split
#X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)

''''
#Construct:
original_dim = 128*128
hidden_encoding_dim = 512
encoding_dim = 64
hidden_decoding_dim = 512
dropout_level = 0.1
(encoder_, decoder_, autoencoder_) = model(original_dim, hidden_encoding_dim, encoding_dim, dropout_level, hidden_decoding_dim)

# Compile the model
autoencoder_.compile(optimizer='adam', loss='binary_crossentropy')

# Fit the model
autoencoder_.fit(X_train, X_train,
                epochs=300,
                batch_size=64,
                shuffle=True,
                validation_data=(X_test, X_test))

# Plot reconstruction
encoded_imgs = encoder_.predict(X_test)
decoded_imgs = decoder_.predict(encoded_imgs)
N = 2  # How many faces we will display
plot_images(1, decoded_imgs)
plot_images(9, decoded_imgs)
''''

input_shape = (128,128,3)
encoded_dim = 1000

input_img = keras.Input(shape=input_shape)
x = keras.layers.Conv2D(128, (3, 3),strides=1,activation='relu', padding='same')(input_img)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Conv2D(64, (3, 3),strides=1,activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128,"relu")(x)
encoded = keras.layers.Dense(encoded_dim,activation='relu')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
x = keras.layers.Dense(128,"relu")(encoded)
x = keras.layers.Dense(4*4*8,"relu")(x)
x = keras.layers.Reshape((4,4,8))(x)
x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
encoder = keras.Model(input_img, encoded)
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy')

#X_train = X_train.reshape(-1,128,128,3)
#X_test = X_test.reshape(-1,128,128,3)
for i in range(len(data_flow)):
''''autoencoder.fit(X_train, X_train,
                epochs=100,
                batch_size=32,
                shuffle=True,
                validation_data=(X_test, X_test))''''
    N_EPOCHS = 100
    BATCH_SIZE = 20
    autoencoder.fit(data_flow[i],
                        shuffle=True,
                        epochs = N_EPOCHS,
                        initial_epoch = 0,
                        steps_per_epoch=NUM_IMAGES / BATCH_SIZE)


# saving in json format
json_model = autoencoder.to_json()
json_file = open('autoencoder_json.json', 'w')
json_file.write(json_model)
json_modelenc = encoder.to_json()
json_file = open('encoder_json.json', 'w')
json_file.write(json_modelenc)
json_modeldec = decoder.to_json()
json_file = open('decoder_json.json', 'w')
json_file.write(json_modeldec)
#encoder.predict(X)   #To get the activations

#decoder.predict()    #To get generate the new faces
