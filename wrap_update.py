import pandas as pd
import numpy as np
from lenskit import batch, topn
from lenskit import crossfold as xf
from lenskit.algorithms import funksvd
from lenskit.algorithms import als 
from lenskit.algorithms import item_knn as knn
from lenskit.matrix import sparse_ratings, CSR

import random
import time
from itertools import product
import util

from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.linalg import lu, solve

from wrap_offlineModel import offlineMFModel, offlinePrediction
import generateLiveRating as gLR
from scipy.spatial.distance import cosine


class liveFunction:
    def newUser_rating_noBias(ratings_newUser, items, gbias, ibias):
        # get the index of items rated by newUser
        # print(ratings_newUser.shape, "items rated by this user")
        idx_rated_items_newUser = items.get_indexer(ratings_newUser['item'])
        
        ubias_newUser = np.mean(ratings_newUser['rating']) - gbias
            # single value
        # print("user bias is ", ubias_newUser)
        ibias_newUser = ibias[idx_rated_items_newUser]
        #print('ibias type', type(ibias_newUser))
            # numpy.ndarray
        #print(len(ibias_newUser))
        '''
        # testing
        print(type(ratings_newUser['rating'].values))
            # numpy.ndarray
        print(type(ratings_newUser['rating']))
            # pandas.core.series.Series
        '''

        ratings_newUser_values = ratings_newUser['rating'].values
        ratings_noBias_newUser = ratings_newUser_values - gbias - ubias_newUser - ibias_newUser
        #print(type(ratings_noBias_newUser))
            # numpy.ndarray
            
        '''
        # testing
        ratings_noBias_newUser = ratings_newUser['rating'] - gbias - ubias_newUser - ibias_newUser
        print(type(ratings_noBias_newUser))
            # # pandas.core.series.Series
        '''
        num_items = len(items)
        ratings_noBias_newUser_match_allItems = np.zeros((1, num_items))
            # ndarray
        # print(ratings_noBias_newUser_match_allItems.size)
            # 1682
        ratings_noBias_newUser_match_allItems[0][idx_rated_items_newUser] = ratings_noBias_newUser
        
        return ratings_noBias_newUser_match_allItems, ubias_newUser
        
    def newUser_predicting(ratings_newUser, newUserID, items, imat, gbias, ibias):
        '''
        --------------- LU decomposition -------------
        # to solve Ax = b (A is square, and somewhat static/fixed, b is more dynamic(frequently changed)
        # A = LU => LUx = b =>  Ly = b, Ux = y => solve y then solve x (Solution!)

        # with PLU decomposition: A = PLU
        # to solve Qp = r
        # convert to Ax = b, here A = Q'Q (square matrix), x = p, b = Q'r;
        
        return:
            a dataframe with ['user', 'item', 'rating', 'prediction']
        '''
    
        ratings_noBias_newUser, ubias_newUser = liveFunction.newUser_rating_noBias(ratings_newUser, 
                                                                                    items, gbias, ibias)
            # ratings_noBias_newUser: numpy.ndarray, (1, numitems)
            # ubias_newUser: single value
        A = np.dot(np.transpose(imat), imat)
        b = np.dot(np.transpose(imat), np.transpose(ratings_noBias_newUser))
        [P, L, U] = lu(A) 
            # P is a permutation matrix, rather than the user latent feature matrix
        y = solve(P.dot(L), b)
        x = solve(U, y)
        # -------------- End of LU decomposition ------

        feature_newUser = np.transpose(x)
            # feature_newUser: numpy.ndarray
        r_predict_noBias = np.dot(feature_newUser, np.transpose(imat))
        # ==============================

        r_predict = r_predict_noBias[0] + gbias + ubias_newUser + ibias
            # the order of corresponding items corresponds to 'itmes'
            
        #===> match r_predict to items
        num_items = len(items)
        ratings_newUser_values_match_allItmes = np.zeros((1, num_items))
        ratings_newUser_extend_all_items = ratings_newUser['rating']
        
        ratings_newUser_values = ratings_newUser['rating'].values
        ## indexing checked!!!
        idx_rated_items_newUser = items.get_indexer(ratings_newUser['item'])
            # get the indices (from index 'items') of values in ratings_newUser['item']
        ratings_newUser_values_match_allItmes[0][idx_rated_items_newUser] = ratings_newUser_values
        d = {'user': newUserID, 'item': items.values, 'rating': ratings_newUser_values_match_allItmes[0], 
            'prediction': r_predict}
        predicted_newUser = pd.DataFrame(d)
            # ['user', 'item', 'rating', 'prediction']
            # with one single user and all items in the offline item set
            # feature_newUser: numpy.ndarray
            
        return predicted_newUser, feature_newUser
                

            
    def computeRMSE(ratings_newUser, predicted_newUser):
        idx_rated_items_newUser = items.get_indexer(ratings_newUser['item'])
        ratings_newUser_values = ratings_newUser['rating'].values
        print('\nActual ratings:')
        print(ratings_newUser_values)
        r_predict_ratedItems = predicted_newUser['prediction'].values[idx_rated_items_newUser]
        print('\nPredicted ratings')
        np.set_printoptions(formatter={'float': lambda x: "{0:0.0f}".format(x)})
        print(r_predict_ratedItems)
        rmse = sqrt(mean_squared_error(ratings_newUser_values, r_predict_ratedItems))
        print('\nRMSE for the prediction of the newUser\'s ratings:', rmse)
        print()

    def relative_ave(singleUser_prediction, items_prediction_ave):
        singleUser_predictionAve = pd.merge(singleUser_prediction, items_prediction_ave, how = 'left', on = ['item'])
            # singleUser_prediction: ['user', 'item', 'rating', 'prediction']
            # items_prediction_ave: ['item', 'ave']
        singleUser_predictionAve['diff'] = singleUser_predictionAve['ave'] - singleUser_predictionAve['prediction']

        return singleUser_predictionAve
    
    def similarity_user_features(umat, users, feature_newUser, method = 'cosine'):
        '''
            ALS has already pre-weighted the user features/item features
            Use either the Cosine distance or the Eculidean distance
        '''        
        nrows, ncols = umat.shape
        # distance = np.zeros([1, nrows])
        distance = []
        if method == 'cosine':
            for i in range(nrows):
                feature_oneUser = umat[i,]
                dis = cosine(feature_oneUser, feature_newUser)
                distance.append(dis)
        elif method == 'eculidean':
            for i in range(nrows):
                feature_oneUser = umat[i,]
                dis = np.linalg.norm(feature_oneUser-feature_newUser)
                    # This works because Euclidean distance is l2 norm and 
                    # the default value of ord parameter in numpy.linalg.norm is 2.
                distance.append(dis)
        # convert to a dataframe with indexing of items
        # print(users)
            # Int64Index
        distance = pd.DataFrame({'user': users.values, 'distance': distance})
        distance.index = users
        # print(distance)
            # 942, 2
        
        return distance
            
 

if __name__ == '__main__':
    ratingsFull = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    newUserID = 377
    ratings, ratings_newUser =  gLR.split_offline_dataset(ratingsFull, newUserID)
    num_features = 50
    algo = als.BiasedMF(num_features)
    path = "./testingPath/"
    filename_prefix = 'savedModel_'
    file_num = 774
    ## compute off line model, run once is good
    filename = filename_prefix + str(file_num)
    [items, imat, gbias, ibias, umat, users, ubias] = offlineMFModel.UIfeatures(ratings, algo)
#    offlineMFModel.save_model(path, items, imat, gbias, ibias, umat, users, ubias, filename)
    ## load model
    
    #===> testing newUser_predicting()
    full_path = path + filename_prefix + str(file_num) + '.npz'
    [items, imat, gbias, ibias, umat, users, _] = offlineMFModel.load_model(full_path)
    [predicted_newUser, feature_newUser] = liveFunction.newUser_predicting(
                                                ratings_newUser, newUserID, items, imat, gbias, ibias)
    #===> testing computeRMSE()
    liveFunction.computeRMSE(ratings_newUser, predicted_newUser)
    
    #===> testing relative_ave()
    [prediction_newUser,_] = liveFunction.newUser_predicting(ratings_newUser, newUserID, items, imat, gbias, ibias)
    offline_item_ave_full_path = "./offline_prediction_testing/items_prediction_ave.npz"
    attri_name = ['item', 'ave']
    items_prediction_ave = offlinePrediction.load_dataset(offline_item_ave_full_path, attri_name)
    items_prediction_ave = items_prediction_ave.astype({attri_name[0]: np.int64})
    singleUser_predictionAve = liveFunction.relative_ave(prediction_newUser, items_prediction_ave)
    # print(singleUser_predictionAve.columns)
    print(singleUser_predictionAve.head(5))
    
    #===> testing similarity_user_features
    full_path_MFmodel = './offline_MFmodel_testing/testingMF_774.npz'
    distance_choice = 'cosine'
    similarity = liveFunction.similarity_user_features(umat, users, feature_newUser, distance_choice)
    print('\nSimilarity the of newUser and other users in offline user set with size: ', similarity.shape)
        # 942, 2