'''
	Train offline models on trainset/offline dataset
	Save the offline models
'''
import time
import pandas as pd
import numpy as np
import csv
from dataset_split import datasetSplit
from lenskit.algorithms import als 
from wrap_offlineModel import offlineMFModel, offlinePrediction, offlineCalculation
from resampleOffline import train_offline_resample_models
from wrap_update import liveFunction



if __name__ == '__main__':
    
    start = time.time()
  #===> import offline ratings
    # test on ml-100k
    
    '''
    fullpath_ratings = './ml-100k/u.data'
    delimeter = '\t'
    attri_name = ['user', 'item', 'rating', 'timestamp']
    ratings = pd.read_csv(fullpath_ratings, sep = delimeter, names = attri_name)
    # print(ratings.head(5))
    '''
    
    # test on ml-25m
    # fullpath_ratings = '../data/ml-LS_ratings_ready.npz'
        # works
    fullpath_ratings = '../data/ml-LS_ratings_ready.npz'
    attri_name = ['user', 'item', 'rating', 'timestamp']
    ratings = offlinePrediction.load_dataset(fullpath_ratings, attri_name)
    ratings = ratings.astype({'user': int, 'item': int, 'rating': float, 'timestamp': int})
    # print(ratings.head(5))
        # Checked 
    # print(type(ratings['user'][3]))
    # print(type(ratings['item'][3]))
    # print(type(ratings['rating'][3]))
    # print(type(ratings['timestamp'][3]))
        # <class 'numpy.int64'>
        # <class 'numpy.int64'>
        # <class 'numpy.float64'>
        # <class 'numpy.int64'>
    timemark1 = time.time()
    spent = timemark1 - start
    print('Dataset imported!\t%.2f seconds passed for this step.\n' % spent)
    
  #===> split the offline ratings dataset
    fullpath_train = './data_dummy/train.npz'
    fullpath_test = './data_dummy/test.npz'
    [ratings_train, ratings_test] = datasetSplit.split_dataset(ratings)

    #===> save the trainset & testset
    # Run once is good 
    offlinePrediction.save_dataset(fullpath_train, ratings_train)    
    offlinePrediction.save_dataset(fullpath_test, ratings_test)
    
    #===> load trainset
    trainset = datasetSplit.load_data(fullpath_train, attri_name)
    timemark2 = time.time()
    spent = timemark2 - timemark1
    print('Train dataset saved!\t%.2f seconds passed for this step.\n' % spent)

  #===> train save offline MF model with the trainset
    num_features = 50
    algo = als.BiasedMF(num_features)
    [items, imat, gbias, ibias, umat, users, ubias] = offlineMFModel.UIfeatures(trainset, algo)    
    
    #===> save offline MF model
    # run once is good
    path_MF_model = './data_dummy/'
    filename_MF_model = 'MFmodel'
    offlineMFModel.save_model(path_MF_model, items, imat, gbias, ibias, umat, users, ubias, filename_MF_model)
 
    #===> load MF model
    fullpath_MF_model = path_MF_model + filename_MF_model + '.npz'
    [items, imat, gbias, ibias, umat, users, ubias] = offlineMFModel.load_model(fullpath_MF_model)
    timemark3 = time.time()
    spent = timemark3 - timemark2
    print('Main MF models saved!\t%.2f seconds passed for this step.\n' % spent)

  #===> get all possible user-item pairs ['user', 'item']  
    full_UI_pair = offlinePrediction.full_offline_UIpairs(trainset)
    # print('\t done!')
    
    #===> save all possible user-item pairs ['user', 'item']
    # run once is good
    fullpath_full_UI_pair = './data_dummy/full_UI_pair.npz'
    offlinePrediction.save_dataset(fullpath_full_UI_pair, full_UI_pair)
    
    # print('\t saved!')
    #===> load all possible user-item pairs ['user', 'item']
    attri_name = ['user', 'item']
    load_dataset = offlinePrediction.load_dataset(fullpath_full_UI_pair, attri_name)
    timemark4 = time.time()
    spent = timemark3 - timemark3
    print('Dataset full_UI_pair saved!\t%.2f seconds passed for this step.\n' % spent)
    
  #===> get full predictions for all possible user-item pairs with users and items from the trainset
    attri_name = ['user', 'item', 'prediction', 'rating', 'timestamp']
    prediction_merge_ratings = offlinePrediction.predictor(algo, ratings, ratings, attri_name)
    
    #===> save full predictions
    # run once is good
    fullpath_full_prediction = './data_dummy/prediction_full_UI_pair.npz'
    offlinePrediction.save_dataset(fullpath_full_prediction, prediction_merge_ratings)
    
    #===> load full predictions
    load_dataset = offlinePrediction.load_dataset(fullpath_full_prediction, attri_name)
    timemark5 = time.time()
    spent = timemark5 - timemark4    
    print('Offline full prediction saved!\t%.2f seconds passed for this step.\n' % spent)

  #===> get ave predictions for all items from trainset
    items_prediction_ave = offlineCalculation.item_calculate_ave_prediction(prediction_merge_ratings)
        # ['item', 'ave']

    #===> save ave predictions
    # run once is good
    fullpath_items_prediction_ave = './data_dummy/items_prediction_ave.npz'
    offlinePrediction.save_dataset(fullpath_items_prediction_ave, items_prediction_ave)
    
    #===> load ave predictions 
    attri_name = ['item', 'ave']
    load_dataset = offlinePrediction.load_dataset(fullpath_items_prediction_ave, attri_name)
    timemark6 = time.time()
    spent = timemark6 - timemark5
    print('Items average prediction saved!\t%.2f seconds passed for this step.\n' % spent)

  #===> get ratings count for all items from trainset
    items_rating_count = offlineCalculation.item_count_ratings(ratings)
        # ['item', 'count']
    
    #===> save ratings count
    # run once is good
    fullpath_items_rating_count = './data_dummy/items_rating_count.npz'
    offlinePrediction.save_dataset(fullpath_items_rating_count, items_rating_count)

    #===> load ratings count
    attri_name = ['item', 'count']
    load_dataset = offlinePrediction.load_dataset(fullpath_items_rating_count, attri_name)
    timemark7 = time.time()
    spent = timemark7 - timemark6
    print('Items rating count saved!\t%.2f seconds passed for this step.\n' % spent)

  #===> train & save offline resampling models at the same time
    # load the saved models when call the live function
    print('\tStart training & saving the resampled MF models:')
    num_resampling = 20
    alpha = 0.5
    path_resampled_models = './data_dummy/resampled_models/'
    filename_prefix = 'resampled_model_'
    saved_model_destination = train_offline_resample_models(ratings, num_resampling, alpha, algo, path_resampled_models, filename_prefix)
        # 20 fullpaths of the saved resampling models
        # unused returns
    timemark8 = time.time()
    spent = timemark8 - timemark7
    print('Resampled MF models saved!\t%.2f seconds passed for this step.\n' % spent)    
        
    print('Total spent time on training all offline models(in seconds): %0.2f\n' % (time.time() - start))
    
    
    
    
    
    
    
    
    
    
    
    