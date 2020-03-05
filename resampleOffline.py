'''
Save 20 offline model's imat, gbias, ibias
'''
import pandas as pd
import numpy as np
from lenskit import batch, topn
from lenskit import crossfold as xf
from lenskit.algorithms import funksvd
from lenskit.algorithms import als 
from lenskit.algorithms import item_knn as knn
import random
import time
from itertools import product
import statistics

from wrap_offlineModel import offlineMFModel
import generateLiveRating as gLR


def train_offline_resample_models(ratings, numRepetition, alpha, algo, path, filename_prefix):
    numObservations = ratings.shape[0]
    destination_list = []
    for i in range(numRepetition):
        print ('\t\tTraining sample', i+1, ':', end = ' ')
        sampled_ratings = ratings.sample(n = int(numObservations * alpha), replace = False)
        # insert the new user here? 
        # seams not ...
        
        [items, imat, gbias, ibias, umat, users, ubias] = offlineMFModel.UIfeatures(sampled_ratings, algo)
        iteration = i+1 # to name different saved model
        # destination = "destination" + str(iteration)
        filename = filename_prefix + str(iteration)
        destination = offlineMFModel.save_model(path, items, imat, gbias, ibias, umat, users, ubias, filename)
        # save_model(imat, item_num, gbias, ibias, iteration)
        destination_list.append(destination)
            # used to load the saved_model with np.load(destination)
            # all fullpaths of the saved resampling models
    return destination_list


if __name__ == '__main__':

    '''
    # train the whole offline dataset
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    # read ratings of a live user ...


    '''
    
    ## new user data from the same offline dataset
    ratingsFull = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    newUserID = 377
    ratings, ratings_newUser =  gLR.split_offline_dataset(ratingsFull, newUserID)
    
    
    algo = als.BiasedMF(50) # with randomness   
    # resample algorithms
    # repetitions
    m = 20 
    # proportion of Rui to be re sampled each time
    alpha = 0.5
    path = "./offline_resampled_model_testing/"
    filename_prefix = 'model_'
    saved_model_destination = train_offline_resample_models(ratings, m, alpha, algo, path, filename_prefix)
    # print(saved_model_destination)
