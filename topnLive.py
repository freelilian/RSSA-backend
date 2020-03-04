'''
Live function for - Traditional TopN
'''

import pandas as pd
import numpy as np
import time

import generateLiveRating as gLR
from wrap_offlineModel import offlineMFModel
from wrap_update import liveFunction

def descending_prediction(full_path_MFmodel, ratings_newUser, newUserID):
    '''
    '''
    # load trained model:
    [items, imat, gbias, ibias, _, _, _] = offlineMFModel.load_model(full_path_MFmodel)
    [predicted_newUser, _] = liveFunction.newUser_predicting(ratings_newUser, newUserID, items, imat, gbias, ibias)
        #return is a dataframe with ['user', 'item', 'rating', 'prediction']
    prediction_sortedByPrediction = predicted_newUser.sort_values(by = 'prediction', ascending = False)
    
    return prediction_sortedByPrediction



if __name__ == '__main__':

    '''
    alternating
    # train the whole offline dataset
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    '''
    start = time.time()
    # new user data from the same offline dataset
    path_ratings = 'ml-100k/u.data'
    delimiter = '\t'
    newUserID = 377
    ratings_newUser= gLR.new_user_data(path_ratings, delimiter, newUserID)
    
    full_path_MFmodel = './offline_MFmodel_testing/testingMF_377.npz'
    prediction_sortedByPrediction = descending_prediction(full_path_MFmodel, ratings_newUser, newUserID)
    rated_itemID = ratings_newUser['item'].values
    prediction_sortedByPrediction_unrated = prediction_sortedByPrediction[~prediction_sortedByPrediction['item'].isin(rated_itemID)]
    
    print(prediction_sortedByPrediction_unrated.head(10))
    print('spent time in seconds: %0.2f' % (time.time() - start))
    
