
#decodeur=np.save(".../decodeur.h5",x[0])

#allow_pickle=True

#import h5py
#decodeur_model = h5py.File('./decodeur.h5')
#a=list(decodeur_model.attrs)
#print(a)
#import json
#b=json.loads(decodeur_model.attrs['model_config'])
#c=json.loads(decodeur_model.attrs['backend'])
#d= json.loads(decodeur_model.attrs[ 'keras_version'])
#e= json.loads(decodeur_model.attrs[ 'training_config'])

#input 1000 dimension encoded vector

#For the sake of this sprint, we are going to create a random population of 20 vectors
#Each vector is populated with random ramples form a uniform distribution
# population=[]
# for i in range(20):
#     population.append(np.random.rand(1,20))
#print(population[1].shape)

#Next, we will maximise the distance between the first sample that we'll show the victim


def initial_sample(pop, sample_size):
    """This function allows to select the first 10 pictures that are going to be shown to the victim in round 1, while maximising the difference between them.
    The selection is based on the euclidienne distance between the pictures of our initial population.
    The <sample_size> vectors whith the highest distance with the other points are selected.

    Args :
        pop: an array of np.arrays each corresponding with a picture
        sample_size (int): the number of pictures that will be selected for round 1

    Returns :
        np.array containing <sample_size> vectors from encoded pictures


    """
    sample=[]
    distances=np.zeros((len(pop), len(pop)))
    for i in range (len(pop)):
        for j in range (i, len(pop)):
            distances[i,j]=np.linalg.norm(pop[i]-pop[j]) #calcul la distance euclidienne entre 2 vecteurs
    sum_dist_row=distances.sum(axis=1)
    sum_dist_column=distances.sum(axis=0)
    sum_dist=sum_dist_row+sum_dist_column
    #print(sum_dist)
    index=np.argpartition(sum_dist,-sample_size)[-sample_size:] # index des 10 plus grandes distances
    #print(index)
    #print(sum_dist[index])
    for i in range(sample_size):
        sample.append(pop[index[i]])
    return np.asarray(sample)






    # im_rgb = cv2.cvtColor(255*sample_decoded_imgs[i], cv2.COLOR_BGR2RGB)
    # # min_val,max_val=img.min(),img.max()
    # # img = 255.0*(img - min_val)/(max_val - min_val)
    # # img = img.astype(np.uint8)
    # file='parents/parent'+str(i)+'.png'
    # cv2.imwrite(file,im_rgb)
#plt.show()


#evolutionary strategies are for small population (not cross-over but gaussian distribution)


def new_population (pop, parent, lambda_) :
    """ This function allows to mutate the parent's attributes using Gaussian distribution.
        It returns a new population of mutated vectors while keeping the parent.

        Args :
            parent: the array selected by the user
            lambda_ (int): the size of the total population (children + parent)

        Returns :
            array containing <lambda> vectors from encoded pictures

        Example :
            >>> len(new_population(population[0], 4))
            4
            >>> population[0] in new_population(population[0], 4)
            True


    """
    n_children = lambda_ -1 #lambda size of population
    children=[parent]
    std=pop.std(axis=0)          # on a besoin de la dernière population pour obtenir les std

    j=0
    while j<n_children :
        child=parent.copy()
        #print(j)
        while np.linalg.norm(child-parent)<11:   #tant que la distance entre l'enfant et le parent ne soit pas supérieure à 11
            #print(np.linalg.norm(child-parent))
            #print("OK")
            random_value=np.random.normal(0,1)  #pour chaque enfant on choisi alpha
            #max_std= np.argpartition(std,300)[:300] #pour chaque enfant on choisi les 500 index dont la std sont les plus petites
            for i in range(len(child)) :
                child[i]+=random_value*std[i]*0.5

            #sigma=alpha*sigmaofneuron (the standard deviation of the neuron that the encoder returns)
            #because we have neurons at 0 so if they have values it will generate unrealistic faces

        #print(child)
        k=1
        while k<len(children) and  np.linalg.norm(child-children[k])>4: #distance entre les autres enfants supérieur à 4
            #print("OKOK")
            k+=1


        if k==len(children):
            children.append(child)
            #print(np.linalg.norm(child-parent))
            j+=1

    return np.asarray(children)




if __name__=="__main__":

    #before the initial sample
    import numpy as np                   # advanced math library
    import matplotlib.pyplot as plt      # plotting routines
    import random
    #import tensorflow as tf
    #from tensorflow import keras
    #import h5py
    import os
    #import cv2
    from PIL import Image
    import scipy.misc

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # UPLOAD THE DECODER :
    from keras.models import load_model

    sample_size=10
    ##

    import doctest
    doctest.testmod(verbose=True)

    decoder = load_model("decodeur.h5")
    encoded_imgs=np.load("img_female_old_straight.csv.npy")

    pop=initial_sample(encoded_imgs, sample_size) #encoded_imgs from the database which
    #print(pop.shape)



## Montrer la population initiale
    plt.figure(figsize=(20, 4))
    sample_decoded_imgs = decoder.predict(pop)
    for i in range (len(pop)):
        ax = plt.subplot(1, len(pop), i + 1 )
        plt.imshow(sample_decoded_imgs[i].reshape(128,128,3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


## Montrer enfants
    new_pop=new_population(pop, pop[0],4) #pop is the database choose by the user
    children_decoded_imgs = decoder.predict(new_pop)
    print(type(children_decoded_imgs))
    for i in range (len(new_pop)):
        ax = plt.subplot(1, len(new_pop), i + 1 )
        plt.imshow(children_decoded_imgs[i].reshape(128,128,3))
        #name='children/child'+str(i)+'.png'
        #im_rgb = cv2.cvtColor(255*children_decoded_imgs[i], cv2.COLOR_BGR2RGB)

        #Image.fromarray(im_rgb).save()
        #cv2.imwrite(name, im_rgb)
    plt.show()
