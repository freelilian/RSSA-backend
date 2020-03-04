'''
Live function for - Things we think you will hate
'''

import pandas as pd
import numpy as np
import time

import generateLiveRating as gLR
from wrap_update import liveFunction
from wrap_offlineModel import offlineMFModel, offlinePrediction, offlineCalculation


def descending_distance_from_ave(full_path_MFmodel, ratings_newUser, newUserID, 
                                    offline_item_ave_full_path, attri_name):
    '''
    
    '''
    # load trained model:
    [items, imat, gbias, ibias, _, _, _] = offlineMFModel.load_model(full_path_MFmodel)
    
    # predict for a new user
    [prediction_newUser,_] = liveFunction.newUser_predicting(ratings_newUser, newUserID, items, imat, gbias, ibias)
        #return is a dataframe with ['user', 'item', 'rating', 'prediction']
    
    # load item_prediction_ave
    items_prediction_ave = offlinePrediction.load_dataset(offline_item_ave_full_path, attri_name)
    # print(type(items_prediction_ave['item'][3]))
        # np.float64
        
    # convert the type of 'item' to int
    # items_prediction_ave['item'] = pd.to_numeric(items_prediction_ave['item'], downcast='integer')
    # items_prediction_ave = items_prediction_ave.astype({'item': int, 'ave': np.float64})
    items_prediction_ave = items_prediction_ave.astype({attri_name[0]: np.int64})
        # int item_id
    # print(items_prediction_ave.head(10))
    
    newUser_predictionAve = liveFunction.relative_ave(prediction_newUser, items_prediction_ave)
    newUser_predictionAve_sortedByDiff = newUser_predictionAve.sort_values(by='diff', ascending = False)
        # [user, item, rating, prediction, ave, diff]
        # Should be TopN(avePrediction - newUserPrediction)

    return newUser_predictionAve_sortedByDiff
    

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
    offline_item_ave_full_path = "./offline_prediction_testing/items_prediction_ave.npz"
    attri_name = ['item', 'ave']
    newUser_predictionAve_sortedByDiff = descending_distance_from_ave(
                                            full_path_MFmodel, ratings_newUser, newUserID, 
                                            offline_item_ave_full_path, attri_name)
    rated_itemID = ratings_newUser['item'].values
    newUser_predictionAve_sortedByDiff_unrated  = newUser_predictionAve_sortedByDiff[~newUser_predictionAve_sortedByDiff['item'].isin(rated_itemID)]    
    print(newUser_predictionAve_sortedByDiff_unrated.head(10))
    print('spent time in seconds: %0.2f' % (time.time() - start))