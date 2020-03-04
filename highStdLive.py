'''
Live function for - Things we have no clue about
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
from os import listdir
from os.path import isfile, join

from wrap_offlineModel import offlineMFModel
from wrap_update import liveFunction
import generateLiveRating as gLR

def load_resampled_model_path(path_resampled_model):
    offline_model_full_path = []
    for f in listdir(path_resampled_model):
        if isfile(join(path_resampled_model, f)):
            full_path = join(path_resampled_model, f)
                # full_path is the full path of each npz file
            #print(full_path)
            offline_model_full_path.append(full_path)
    return offline_model_full_path
    
    
def live_user_prediction_resampling(path_resampled_model, ratings_newUser, newUserID):
    # predict for the newUser with 20 resampled offline dataset (partial dataset)
    offline_model_full_path = load_resampled_model_path(path_resampled_model)
    resampled_prediction_dataFrames = []
    for full_path in offline_model_full_path:
        [resampled_items, resampled_imat, resampled_gbias, resampled_ibias, _, _, _] = offlineMFModel.load_model(full_path)
        [resampled_predicted_newUser, _] = liveFunction.newUser_predicting(ratings_newUser, newUserID, resampled_items, resampled_imat, resampled_gbias, resampled_ibias)
            # resampled_predicted_newUser is dataframe with ['user', 'item', 'rating', 'prediction']
        resampled_prediction_dataFrames.append(resampled_predicted_newUser)
            # hold all returned dataframe, each element is a dataframe

    return resampled_prediction_dataFrames
    
def concatenate_prediction(resampled_prediction_dataFrames):
    all_resampled_prediction = resampled_prediction_dataFrames[0]
    count = 0
    sum = 0
    # print(all_resampled_prediction.shape)
    for p in resampled_prediction_dataFrames:
        count = count + 1
        # print(p.shape, len(p['item']), 'items')
        # sum = sum + p.shape[0]
        if count > 1:
            all_resampled_prediction = all_resampled_prediction.append(p, sort = False)
    # print(sum, all_resampled_prediction.shape)        
            
    return all_resampled_prediction
    
def check_items(all_resampled_prediction):
    '''
    check if all_resampled_prediction contains all items
    JUST for testing
    '''
    # check np.unique() usage in https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
    items, items_inverse, items_count = np.unique(all_resampled_prediction['item'], return_inverse = True, return_counts = True)
        # items[items_inverse] reconstruct the items in the same order in all_resampled_prediction['item']
    #print(items, all_resampled_prediction['item'].unique())
        # numpy.unique() return sorted unique values
        # all_resampled_prediction['item'].unique() return un-sorted values
    print(len(items), np.min(items_count))
        # 1682, 4
        # 1682 are all resampled, the item with minimum resampling was resampled 4 times
    # print(type(items_inverse))
        # numpy.ndarray
    print(items_inverse.size)    

def descending_std(full_path_MFmodel, path_resampled_model, ratings_newUser, newUserID):
    
    #===> predict for the newUser with full offline dataset
    [items, imat, gbias, ibias, _, _, _] = offlineMFModel.load_model(full_path_MFmodel)
    [predicted_newUser, _] = liveFunction.newUser_predicting(ratings_newUser, newUserID, items, imat, gbias, ibias)
        # ['user', 'item', 'rating', 'prediction']
    
    #===> resampleing
    resampled_prediction_dataFrames = live_user_prediction_resampling(path_resampled_model, ratings_newUser, newUserID)
        #20 resampling prediction dataframes 
    all_resampled_prediction = concatenate_prediction(resampled_prediction_dataFrames)
        #predictions of all 20 resampling

    # items, items_inverse, items_count = np.unique(all_resampled_prediction['item'], return_inverse = True, return_counts = True)
    # print(type(items))
        # numpy.ndarray
    # print(len(items))
        # 1682 items
    
    #??? is there a way to calculate the std of one column by labels addressed in another column in python?
    #??? if so, then replace the following for loop
    
    # only iterrate through the unrated items by the newUser
    rated_itemID = ratings_newUser['item'].values
    unrated_items_predicted_newUser = predicted_newUser[~predicted_newUser['item'].isin(rated_itemID)]
    unrated_itemID = unrated_items_predicted_newUser['item'].to_numpy()
    # print(type(unrated_itemID))
        # numpy.ndarray
    # print(len(unrated_itemID))
        # 1625
    
    unrated_items_std = []

    # for index, row in unrated_items_predicted_newUser.iterrows():
    for i in unrated_itemID:
        one_item_resampled = all_resampled_prediction[all_resampled_prediction['item'] == i]
        one_item_resampled_prediction_values = one_item_resampled['prediction'].to_numpy()
        item_std = np.std(one_item_resampled_prediction_values)
        unrated_items_std.append(item_std)

    #np.set_printoptions(formatter={'float': lambda x: "{0:0.0f}".format(x)})
    # print(len(items_std), np.max(items_std), np.min(items_std))
    unrated_items_rating_values = unrated_items_predicted_newUser['rating'].to_numpy()
    unrated_items_prediction_values = unrated_items_predicted_newUser['prediction'].to_numpy()
    unrated_items_std_dataframe = pd.DataFrame({'user': newUserID, 'item': unrated_itemID, 'rating': unrated_items_rating_values, 
                                        'prediction': unrated_items_prediction_values, 'std': unrated_items_std})
    
    # print(type(unrated_items_std_dataframe['std'][3]))
    unrated_items_std_dataframe_sortedByStd = unrated_items_std_dataframe.sort_values(by = 'std', ascending = False)
        # ['user', 'item', 'rating', 'prediction', 'std']
    # print(unrated_items_std_dataframe_sorted.head(10))
    
    return unrated_items_std_dataframe_sortedByStd
    
  
if __name__ == '__main__':
    start = time.time()
    # new user data from the same offline dataset
    path_ratings = 'ml-100k/u.data'
    delimiter = '\t'
    newUserID = 377
    ratings_newUser= gLR.new_user_data(path_ratings, delimiter, newUserID)
    
    full_path_MFmodel = './offline_MFmodel_testing/testingMF_377.npz'
    path_offline_resampled_model = './offline_resampled_model_testing/'
    unrated_items_std_dataframe_sortedByStd = descending_std(full_path_MFmodel, path_offline_resampled_model, ratings_newUser, newUserID)
    print(unrated_items_std_dataframe_sortedByStd.head(10))
         
    #np.set_printoptions(formatter={'float': lambda x: "{0:0.0f}".format(x)})
    print('spent time in seconds: %0.2f' % (time.time() - start))








