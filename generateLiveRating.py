'''
Generate rating data for a new user 
from the offline dataset
'''
import pandas as pd


def split_offline_dataset(ratingsFull, newUserID):
    ratings = ratingsFull[ratingsFull['user'] != newUserID] 
    ratings_newUserID = ratingsFull[ratingsFull['user'] == newUserID]
    ratings_newUser = ratings_newUserID[['user', 'item', 'rating']]
        
    return ratings, ratings_newUser
    
def new_user_data(path_ratings, delimiter, newUserID = 37):
    '''
    Source to read data may differ
    '''
    ratingsFull = pd.read_csv(path_ratings, sep = delimiter, names=['user', 'item', 'rating', 'timestamp'])   
        # should be a valid id included in ratingsFull
    ratings, ratings_newUser =  split_offline_dataset(ratingsFull, newUserID)
    
    return ratings_newUser
    
    
if __name__ == '__main__':    
    path_ratings = 'ml-100k/u.data'
    delimiter = '\t'
    newUserID = 73
    ratings_newUser = new_user_data(path_ratings, delimiter, newUserID)
    print(ratings_newUser.shape)