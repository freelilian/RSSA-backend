import pandas as pd
import numpy as np
from lenskit import batch, topn
from lenskit import crossfold as xf
#from lenskit.algorithms import funksvd
from lenskit.algorithms import als 
from lenskit.matrix import sparse_ratings, CSR, csr_to_scipy
import util
from itertools import product
from numpy import linalg as LA
from scipy.sparse import csr_matrix

class offlineMFModel:
    def UIfeatures(ratings, algo):
        # algo = als.BiasedMF(50)
        model = algo.train(ratings)

        # save model data
        algo.save_model(model, "./model_output/")

        path = util.npz_path("./model_output/.npz")
        with np.load(path) as npz:
            users = pd.Index(npz['users'])
            items = pd.Index(npz['items'])
                # users and items were converted to values, so we need to re-index them when reading the data
                # please refer to mf_common.py line 191
            umat = npz['umat']
            imat = npz['imat']
            gbias = npz['gbias'][0]
            ubias = npz['ubias']
            ibias = npz['ibias']

        print(len(items), ' items in itemset')
            # check if item set is full since 3 users is split from the dataset
            # 1682, full item set
       
        return items, imat, gbias, ibias, umat, users, ubias

    def save_model(path, items, imat, gbias, ibias, umat, users, ubias, filename):
        full_path = path + filename
        np.savez_compressed(full_path, items = items.values, imat = imat, gbias = gbias, ibias = ibias, 
                                        umat = umat, users = users.values, ubias = ubias)
        
        #np.savetxt(path, mergedStd, delimiter=" ")
        return full_path

    def load_model(full_path):
        model_loaded = np.load(full_path)
        items = pd.Index(model_loaded['items'])
        imat = model_loaded['imat']
        gbias = model_loaded['gbias']
        ibias = model_loaded['ibias']
        umat = model_loaded['umat']
        users = pd.Index(model_loaded['users'])
        ubias = model_loaded['ubias']
        
        return items, imat, gbias, ibias, umat, users, ubias

class offlinePrediction:
    '''
    prediction done for the offline dataset(all possible user-item pair, that is fill in the whole UI matrix)
    '''
    def full_offline_UIpairs(ratings):
        '''
        extract unique userset and itemset
        generate full dataset with all possible user-item pairs, i.e. fill the UI matrix in long format
        '''
        userset = np.unique(ratings['user']) # numpfy.ndarray
        itemset = np.unique(ratings['item']) # numpy.ndarray
        all_UI_pair = product(userset, itemset)
        all_UI_pair = pd.DataFrame(all_UI_pair)
        all_UI_pair.columns = ['user', 'item']

        return all_UI_pair
    
    def save_dataset(full_path, dataset):
        '''
            save single dataset as .npz file
        '''
        np.savez_compressed(full_path, dataset = dataset)
            
    def load_dataset(full_path, attri_name):
        '''
            load single dataset from saved .npz file
        '''
        model_loaded = np.load(full_path)
        data = model_loaded['dataset']
            # numpy.ndarray
        dataset = pd.DataFrame(data, columns = attri_name)
            # dataframe
        return dataset
        
    def predictor(algo, ratings, ratings_trainset, new_attri_name):
        '''
        return a dataframe with all UI pairs, ['user', 'item', 'prediction', 'rating', 'timestamp']
        '''
        model = algo.train(ratings_trainset)
        all_UI_pair = offlinePrediction.full_offline_UIpairs(ratings)
        prediction_all_UI_pair = batch.predict(algo, all_UI_pair, model)
            # https://lkpy.readthedocs.io/en/latest/batch.ht ml?highlight=batch
            # batch.predict returns dataframe [dataPairs['all columns'], 'prediction']
            # ['user', 'item', 'prediction'] for all_UI_pair
        merged_all_UI_pair = offlinePrediction.merge_prediction_rating(ratings, prediction_all_UI_pair, new_attri_name)
            # [new_attri_name]
            # usually: ['user', 'item', 'prediction', 'rating', 'timestamp']
        
        return merged_all_UI_pair
        
    def merge_prediction_rating(ratings, prediction_all_UI_pair, attri_name):
        merged_all_UI_pair = pd.merge(prediction_all_UI_pair, ratings, how = 'left', on = ['user', 'item'])
        merged_all_UI_pair.columns = attri_name
        
        return merged_all_UI_pair 
        

class offlineCalculation:
    '''
    calculate prediction ave for all items
    calculate distance of a users prediction from the general average prediction
    '''
    def item_calculate_ave_prediction(full_prediction):
        '''
        full_prediction: a dataframe has all UI pairs, with prediction and observed ratings merged:
            ['user', 'item', 'prediction', 'rating', 'timestamp']
        '''
        items_prediction_ave = []
        itemset = np.unique(full_prediction['item']) # numpy.ndarray
        for item in itemset:
            item_predictions = full_prediction[full_prediction['item'] == item]['prediction'].values
                # predictions of one single item from all users
            items_prediction_ave.append(np.mean(item_predictions))
        
        #print(type(itemset))   
        #print(type(items_prediction_ave))
        # items_prediction_ave is a list, do I need to convert it numpy.ndarray using np.asarray(items_prediction_ave)
        item_prediction_ave_dataframe = pd.DataFrame({'item': itemset, 'ave': items_prediction_ave})
            #['item', 'ave']
        #print(type(item_prediction_ave_dataframe['item'][1]))
            #numpy.int64
        #print(type(item_prediction_ave_dataframe['ave'][1]))
            #numpy.float64 
        
        return item_prediction_ave_dataframe
        
   
    def item_count_ratings(ratings):
        #full_prediction_sortedByPrediction = full_prediction.sort_values(by = 'prediction', ascending = False)
        #topN_full_prediction = full_prediction_sortedByPrediction.head(numTopN)
        items_rating_count = []
        itemset = np.unique(ratings['item']) # numpy.ndarray
        for item in itemset:
            item_ratings = ratings[ratings['item'] == item]
            items_rating_count.append(item_ratings.shape[0])
            
        items_rating_count_dataframe = pd.DataFrame({'item': itemset, 'count': items_rating_count})
            # ['item' ,'count']
        
        return items_rating_count_dataframe
            
        
    def singular_value_weight(ratings, users):
        rmat, users, items = sparse_ratings(ratings)
            # refer to lenskit.matrix， line43-70
        num_rows = len(users.values)
        num_cols = len(items.values)
        # print(len(rmat.values))
        # print(len(rmat.colinds))
        # print(len(rmat.rowptrs))
        UI_mat= csr_matrix( (rmat.values, rmat.colinds, rmat.rowptrs), 
                        shape = (num_rows, num_cols) ).toarray()
        UI_mat_transpost = np.transpose(UI_mat)
            # Non-square matrix has no eigenvalues, 
            # instead, use singular values
        UI_square = UI_mat.dot(UI_mat_transpost)
        # print(UI_mat)
        singular_val = LA.eigvals(UI_square)
        # print(len(singular_val))
            # 942, = num_rows, I.E. num_users
            # numpy.ndarray
        singular_val = pd.DataFrame({'singular': singular_val})
        singular_val.index = users
        
        return singular_val
    
    
if __name__ == '__main__':
    '''
    # alternating: 
    # train the whole offline dataset
    ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    # read ratings of a live user ...
    
    path = "./offline_MFModel/"
    filename = 'mf_model'

    '''

    ########-------------- new user data from the same offline dataset
    
    newUserID = 377
    ratingsFull = pd.read_csv('ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    ratings = ratingsFull[ratingsFull['user'] != newUserID] 
    ratings_newUser = ratingsFull[ratingsFull['user'] == newUserID]
    
    path = "./offline_MFmodel_testing/"
    filename_prefix = 'testingMF_'
    file_num = newUserID
    filename = filename_prefix + str(file_num)
    ## compute offline model, run once is good
    
    
    
    num_features = 50
    algo = als.BiasedMF(num_features)
    items, imat, gbias, ibias, umat, users, ubias = offlineMFModel.UIfeatures(ratings, algo)
    offlineMFModel.save_model(path, items, imat, gbias, ibias, umat, users, ubias, filename)
    

    ## load model
    full_path = path + filename_prefix + str(file_num) + '.npz'
    [items, imat, gbias, ibias, umat, users, ubias] = offlineMFModel.load_model(full_path)
    # print(ubias[0:3])
    
    #===> get predictions for all possible user-item pairs
    # save all possible user-item pairs ['user', 'item']
    # algo = als.BiasedMF(num_features)
    full_UI_pair = offlinePrediction.full_offline_UIpairs(ratings)
    # print(full_UI_pair.shape)
    path = './offline_prediction_testing/'
    filename = 'full_UI_pair'
    full_path = path + filename + '.npz'
    offlinePrediction.save_dataset(full_path, full_UI_pair)
    attri_name = ['user', 'item']
    load_dataset = offlinePrediction.load_dataset(full_path, attri_name)
#     print('\nCheck if full_UI_pair saved and loaded successfull', load_dataset.shape)
#    print(full_UI_pair.head(10))
#    print(load_dataset.head(10))
    
    attri_name = ['user', 'item', 'prediction', 'rating', 'timestamp']
    prediction_merge_ratings = offlinePrediction.predictor(algo, ratings, ratings, attri_name)
    filename = 'prediction_all_UI_pair'
    full_path = path + filename + '.npz'
    offlinePrediction.save_dataset(full_path, prediction_merge_ratings)
    load_dataset = offlinePrediction.load_dataset(full_path, attri_name)
#     print('\nCheck if prediction_merge_ratings saved and loaded successfull', load_dataset.shape)
#    print(prediction_merge_ratings.head(10))
#    print(load_dataset.head(10))
    
    #===>calculate average predictions for all items in dataset 
    # save item average prediction
    items_prediction_ave = offlineCalculation.item_calculate_ave_prediction(prediction_merge_ratings)
        #return [item, ave]
    # print(items_prediction_ave.shape)    
    filename = 'items_prediction_ave'
    full_path = path + filename + '.npz'
    offlinePrediction.save_dataset(full_path, items_prediction_ave)
    attri_name = ['item', 'ave']
    load_dataset = offlinePrediction.load_dataset(full_path, attri_name)
#     print('\nCheck if items_prediction_ave saved and loaded successfull', load_dataset.shape)
#    print(items_prediction_ave.head(10))
#    print(load_dataset.head(10))
    
    #===>calculate average predictions for all items in dataset 
    # save item average prediction
    items_rating_count = offlineCalculation.item_count_ratings(ratings)
        #return [item, count]
    # print(items_rating_count.shape)
    filename = 'items_rating_count'
    full_path = path + filename + '.npz'
    offlinePrediction.save_dataset(full_path, items_rating_count)
    attri_name = ['item', 'count']
    load_dataset = offlinePrediction.load_dataset(full_path, attri_name)
#     print('\nCheck if items_rating_count saved and loaded successfull', load_dataset.shape)
#    print(load_dataset.head(10))
#    print(load_dataset.head(10))
    
    #===>Similarity: singular value to add weights
    '''
    full_path = path + filename_prefix + str(file_num) + '.npz'
    singular_value = offlineCalculation.singular_value_weight(ratings, users)
        # dataframe ['singular'], len = num_users
    filename = 'singular_value'
    offlinePrediction.save_dataset(path, filename, singular_value)
    '''

    
    
    
    