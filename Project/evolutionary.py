import numpy as np                   # advanced math library
#import matplotlib.pyplot as plt      # plotting routines

#input 1000 dimension encoded vector

#For the sake of this sprint, we are going to create a random population of 20 vectors
#Each vector is populated with random ramples form a uniform distribution
population=[]
for i in range(20):
    population.append(np.random.rand(1,20))
#print(population[1].shape)

#Next, we will maximise the distance between the first sample that we'll show the victim
sample_size=10

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

initial_sample(population, sample_size)

def new_children (parent, lambda_) :

    n_children = lambda_ -1 #lambda size of population
    children=[]
    for j in range (n_children) :
        #if np.random.rand(1,1) <1 : propabilité d'avoir notre attribut qui mute
        child=parent.copy()
        for i in range(len(parent)) :
            random_value=np.random.normal(0,1)
            child[i]+=random_value
        #print(child)
        children.append(child)
    return children

print(population[0])
print(new_children(population[0], 4))
