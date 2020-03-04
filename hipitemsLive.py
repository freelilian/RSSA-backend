'''
Live function for - Things you will be among the first to try
'''

import pandas as pd
import numpy as np
import time

from wrap_update import liveFunction
from wrap_offlineModel import offlineMFModel, offlinePrediction, offlineCalculation
import generateLiveRating as gLR



def novel(full_path_MFmodel,ratings_newUser, newUserID, 
            offline_items_rating_count_full_path, attri_name, numTopNprediction = 200):
    '''
    
    '''
    # load trained model:
    [items, imat, gbias, ibias, _, _, _] = offlineMFModel.load_model(full_path_MFmodel)
    
    # predict for a new user
    [predicted_newUser, _] = liveFunction.newUser_predicting(ratings_newUser, newUserID, items, imat, gbias, ibias)
        #return is a dataframe with ['user', 'item', 'rating', 'prediction']
        
    # print(predicted_newUser.shape)
        # [1682, 4]
        
    # load items_rating_count
    items_rating_count = offlinePrediction.load_dataset(offline_items_rating_count_full_path, attri_name)
    items_rating_count = items_rating_count.astype({attri_name[0]: np.int64, attri_name[1]: np.int64})
    
    predicted_newUser_count = pd.merge(predicted_newUser, items_rating_count, how = 'left', on = 'item')
    items_unrated_newUser = predicted_newUser_count[predicted_newUser_count['rating'] < 1]
    # print(items_unrated_newUser.shape[0], ' items unrated')
    items_unrated_newUser_sortedByPrediction = items_unrated_newUser.sort_values(by = 'prediction', ascending = False)
    topNprediction_items_unrated_newUser = items_unrated_newUser_sortedByPrediction.head(numTopNprediction)
    topNprediction_items_unrated_sortedByCount = topNprediction_items_unrated_newUser.sort_values(by = 'count', ascending = True)
        # ['user', 'item', 'rating', 'prediction', 'count']
    # print(topNprediction_items_unrated_sortedByCount.columns)
    
    return topNprediction_items_unrated_sortedByCount
    
    
    

if __name__ == '__main__':
    '''
    '''
    start = time.time()
    # new user data from the same offline dataset
    path_ratings = 'ml-100k/u.data'
    delimiter = '\t'
    newUserID = 377
    ratings_newUser= gLR.new_user_data(path_ratings, delimiter, newUserID)
    
    full_path_MFmodel = './offline_MFmodel_testing/testingMF_377.npz'
    offline_items_rating_count_full_path = "./offline_prediction_testing/items_rating_count.npz"
    attri_name = ['item', 'count']
    numTopNprediction = 200
    
    topNprediction_items_unrated_sortedByCount = novel(full_path_MFmodel,ratings_newUser, newUserID, 
                                                offline_items_rating_count_full_path, attri_name, numTopNprediction)
    print(topNprediction_items_unrated_sortedByCount.head(10))
    print('spent time in seconds: %0.2f' % (time.time() - start))
