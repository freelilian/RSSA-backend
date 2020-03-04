'''
Live function for - Things that are controversial
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

from wrap_update import liveFunction
from wrap_offlineModel import offlineMFModel, offlinePrediction
import generateLiveRating as gLR


def find_neighbors(umat, users, feature_newUser, distance_choice, num_neighbors):
    similarity = liveFunction.similarity_user_features(umat, users, feature_newUser, distance_choice)
        # ['user_id', 'distance']
    # min_distanc = ??
    similarity_sorted = similarity.sort_values(by = 'distance', ascending = True)
    neighbors_similarity = similarity_sorted.head(num_neighbors)
    
    return neighbors_similarity
                                    
def extract_neighbor_predictions(umat, users, feature_newUser, offline_prediction, distance_choice, num_neighbors):
    '''
        Extract the offline prediction of neighbors on all the items in the offline item set
    '''
    #===> extract userIDs of neighbors and the corresponding predictions from the offline prediction
    neighbors_similarity = find_neighbors(umat, users, feature_newUser, distance_choice, num_neighbors)
        # ['user', 'distance']
    neighbors_userID = neighbors_similarity['user'].values
    neighbors_offline_prediction = offline_prediction[offline_prediction['user'].isin(neighbors_userID)]

    return neighbors_offline_prediction
    
    
def variance(full_path_MFmodel, ratings_newUser, newUserID, 
                full_path_offline_prediction, distance_choice, num_neighbors):
    #===> extract unrated itemIDs of the newUser and all prediction ...
        # of the unrated items from neighbors.
    [items, imat, gbias, ibias, umat, users, _] = offlineMFModel.load_model(full_path_MFmodel)
    [predicted_newUser, feature_newUser] = liveFunction.newUser_predicting(ratings_newUser, newUserID, 
                                                                            items, imat, gbias, ibias)
        # ['user', 'item', 'rating', 'prediction']
        # with one single user and all items in the offline item set
        # feature_newUser: numpy.ndarray
        
    attri_name = ['user', 'item', 'prediction', 'rating', 'timestamp']
    offline_prediction = offlinePrediction.load_dataset(full_path_offline_prediction, attri_name)
    offline_prediction = offline_prediction.astype({'user': np.int64, 'item':np.int64})
    neighbors_offline_prediction = extract_neighbor_predictions(umat, users, feature_newUser, 
                                            offline_prediction, distance_choice, num_neighbors)
    
    rated_itemID = ratings_newUser['item'].values
    # all_items = predicted_newUser['item'].to_frame()
    # print('\nSize of all_items:', all_items.shape)
        # (1682, 1))
    unrated_items_predicted_newUser = predicted_newUser[~predicted_newUser['item'].isin(rated_itemID)]
    # print('\nSize of unrated_items_predicted_newUser:', unrated_items_predicted_newUser.shape)
        # (1625, 4)
    # unrated_items_neighbors_offline_prediction = neighbors_offline_prediction[neighbors_offline_prediction['item'].isin(unrated_items.values)]
    
    # The logic is similiar to that in highStdLive.py
    # Calculating in different way.
    # ??? Check which way works less expensive
    unrated_items_var = []
    item_check = []
    for index, row in unrated_items_predicted_newUser.iterrows():
        one_item_neighbors_prediction_values = neighbors_offline_prediction[neighbors_offline_prediction['item'] == row['item']]['prediction'].values
        item_var = np.var(one_item_neighbors_prediction_values)
        unrated_items_var.append(item_var)
        # item_check.append(row['item'])
        
    item_values = unrated_items_predicted_newUser['item'].to_numpy()
    rating_values = unrated_items_predicted_newUser['rating'].to_numpy()
    prediction_values = unrated_items_predicted_newUser['prediction'].to_numpy()
    unrated_items_var_dataframe = pd.DataFrame({'user': newUserID, 'item': item_values, 'rating': rating_values, 'prediction': prediction_values, 'var': unrated_items_var})
    # unrated_items_var_dataframe = pd.DataFrame({'user': newUserID, 'item': item_values, 'rating': rating_values, 'prediction': prediction_values, 'var': unrated_items_var, 'item_check': item_check})

    # ??? check with matching method???????????????????????
    
    # ?????????????????   wired results after sorted
    # unrated_items_var_dataframe['item'] = unrated_items_predicted_newUser['item']
    
    #unrated_items_var_dataframe['item'] = unrated_items_predicted_newUser['item'].to_numpy()
    #unrated_items_var_dataframe['rating'] = unrated_items_predicted_newUser['rating']
    #unrated_items_var_dataframe['prediction'] = unrated_items_predicted_newUser['prediction']

    # print('\nSize of predicted_newUser:', predicted_newUser.shape)
        # (1682, 4)
    # print('\nSize of unrated_items_var_dataframe:', unrated_items_var_dataframe.shape)
  
    unrated_items_var_dataframe_sortByVar = unrated_items_var_dataframe.sort_values(by = 'var', ascending = False)
        # ['user', 'item', 'rating', 'prediction', 'var']
    
    return unrated_items_var_dataframe_sortByVar
    
if __name__ == '__main__':
    '''
    '''
    start = time.time()
    # new user data from the same offline dataset
    path_ratings = 'ml-100k/u.data'
    delimiter = '\t'
    newUserID = 377
    ratings_newUser= gLR.new_user_data(path_ratings, delimiter, newUserID)
    
    file_num = newUserID
    full_path_MFmodel = './offline_MFmodel_testing/testingMF_' + str(file_num) + '.npz'
    full_path_offline_prediction = './offline_prediction_testing/prediction_all_UI_pair.npz'

    # print('\nSize of full offline prediction: ', offline_prediction.shape)
        # (1584444, 5)
    distance_choice = 'eculidean'
    num_neighbors = 20
    unrated_items_var_dataframe_sortByVar = variance(full_path_MFmodel, ratings_newUser, newUserID, 
                                                        full_path_offline_prediction, distance_choice, num_neighbors)
    print(unrated_items_var_dataframe_sortByVar.head(10))

    print('spent time in seconds: %0.2f' % (time.time() - start))
    