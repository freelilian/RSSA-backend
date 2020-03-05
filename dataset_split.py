'''
	Split dataset:
		Use 10% of the users' ratins from offline dataset as the test dataset  
		Use 90% of the users' ratins from offline dataset as the train dataset   
'''
import csv 
import pandas as pd
import numpy as np
from wrap_offlineModel import offlinePrediction
import random

class datasetSplit:
    def split_dataset(ratings, percentage_testUserset = 0.1):
        '''
            ratings: ['user', 'item', 'rating', 'timestamp']
            percentage_testUserset: percentage of users in the user set to be the train dataset
        '''
        ratings_sortByUser = ratings.sort_values(by = 'user', ascending = True)
        userIDs = np.unique(ratings_sortByUser['user'])
        num_users = len(userIDs)
            # sorted
        num_test = int(num_users * percentage_testUserset)
        num_train = num_users - num_test
        userIDs_train = userIDs[0:num_train]
            # does not include userIDs[num_train]
            # np.array
        ratings_train = ratings[ratings['user'].isin(userIDs_train)]
        ratings_test = ratings[~ratings['user'].isin(userIDs_train)]
        # print(ratings_test.shape)
            # (9981, 4)
        # excluding items that are not rated by users in the trainset
        itemset_train = np.unique(ratings_train['item'])
        ratings_test = ratings_test[ratings_test['item'].isin(itemset_train)]
        # print(ratings_test.shape)
            # (9973, 4)
            
        return ratings_train, ratings_test
        
    def load_data(full_path_train, attri_name):
        trainset = offlinePrediction.load_dataset(full_path_train, attri_name)
        trainset = trainset.astype({'user': int, 'item': int})
            # ['user', 'item', 'rating', 'timestamp']
        
        return trainset

    def load_live_user_ratings(full_path_test, attri_name):    
        testset = datasetSplit.load_data(full_path_test, attri_name)
        test_userIDs = np.unique(testset['user'])
            # np.ndarray
        live_userID = random.choice(test_userIDs)
            # np.int64, a single number
        live_userID = list([live_userID])
            # or: live_userID = np.array([live_userID])
            # isin needs an list-like input
        liveUser_ratings = testset[testset['user'].isin(live_userID)]
        # print(liveUser_ratings.shape)
            # (varieNumber, 4)
        return liveUser_ratings, live_userID[0]
        
if __name__ == "__main__":
    
    dataset_full_path = './ml-100k/u.data'
    delimeter = '\t'
    attri_name = ['user', 'item', 'rating', 'timestamp']
    ratings = pd.read_csv(dataset_full_path, sep = delimeter, names = attri_name)
    
    full_path_train = './data_dummy/train.npz'
    full_path_test = './data_dummy/test.npz'
    [ratings_train, ratings_test] = datasetSplit.split_dataset(ratings)
        
    offlinePrediction.save_dataset(full_path_train, ratings_train)    
    offlinePrediction.save_dataset(full_path_test, ratings_test) 
    
    trainset = datasetSplit.load_data(full_path_train, attri_name)
    liveUser_ratings, liveUserID = datasetSplit.load_live_user_ratings(full_path_test, attri_name)
    print(liveUser_ratings.head(5))
    # print(liveUser_ratings.columns)
        # ['user', 'item', 'rating', 'timestamp']
    # print(type(liveUserID))
        # np.int64
    
    