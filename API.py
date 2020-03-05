'''
APIs for all recommendation list
'''

import time
import generateLiveRating as gLR
import topnLive
import hateitemsLive
import hipitemsLive
import highStdLive
import controversialLive

class recommenderRSSA:
    def topn_items(full_path_MFmodel, ratings_newUser, newUserID, numRec = 10):
        start = time.time()
        prediction_sortedByPrediction = topnLive.descending_prediction(full_path_MFmodel, ratings_newUser, newUserID)
        rated_itemID = ratings_newUser['item'].values
        prediction_sortedByPrediction_unrated = prediction_sortedByPrediction[~prediction_sortedByPrediction['item'].isin(rated_itemID)]
        recommendation = prediction_sortedByPrediction_unrated.head(numRec)
        print('\nSpent time in seconds: %0.2f' % (time.time() - start))

        return recommendation
        
    def hate_items(full_path_MFmodel, ratings_newUser, newUserID, 
                    offline_item_ave_full_path, attri_name, numRec = 10):
        start = time.time()
        predictionAve_sortedByDiff = hateitemsLive.descending_distance_from_ave(
                                        full_path_MFmodel, ratings_newUser, newUserID, 
                                        offline_item_ave_full_path, attri_name)
        rated_itemID = ratings_newUser['item'].values
        predictionAve_sortedByDiff_unrated = predictionAve_sortedByDiff[~predictionAve_sortedByDiff['item'].isin(rated_itemID)] 
        recommendation = predictionAve_sortedByDiff_unrated.head(numRec)
        print('\nSpent time in seconds: %0.2f' % (time.time() - start))
        
        return recommendation
        
    def novel_items(full_path_MFmodel, ratings_newUser, newUserID, 
                    offline_items_rating_count_full_path, attri_name, 
                    numTopNprediction, numRec = 10):
        start = time.time()
        items_unrated_sortedByCount = hipitemsLive.novel(full_path_MFmodel,ratings_newUser, newUserID, 
                                                                offline_items_rating_count_full_path,
                                                                attri_name, numTopNprediction)
        recommendation = items_unrated_sortedByCount.head(numRec)
        print('\nSpent time in seconds: %0.2f' % (time.time() - start))
        
        return recommendation
        
    def high_std_items(full_path_MFmodel, path_resampled_model, ratings_newUser, newUserID, numRec = 10):
        start = time.time()
        unrated_items_std_sortedByStd = highStdLive.descending_std(full_path_MFmodel, path_resampled_model, ratings_newUser, newUserID)
        recommendation = unrated_items_std_sortedByStd.head(numRec)
        print('\nSpent time in seconds: %0.2f' % (time.time() - start))
        
        return recommendation   
        
    def controversial_items(full_path_MFmodel, full_path_offline_prediction, ratings_newUser, 
                                newUserID, distance_choice, num_neighbors, numRec = 10):
        start = time.time()
        unrated_items_var_sortByVar = controversialLive.variance(full_path_MFmodel, ratings_newUser, newUserID, 
                                                        full_path_offline_prediction, distance_choice, num_neighbors)
        recommendation = unrated_items_var_sortByVar.head(numRec)            
        print('\nSpent time in seconds: %0.2f' % (time.time() - start))
        
        return recommendation

class printRSSAlist:        
    def print_to_screen(recommendation, userID, list_name, attributes_to_print, formatting_value):
        print("    For userID:", userID)
        print("\t", list_name)
        num_attri = len(attributes_to_print)
        formatting_attri = ""
        for i in range(1, num_attri+1):
            formatting_attri = formatting_attri + "%20s"
        print(formatting_attri % tuple(attributes_to_print))
        
        to_print = recommendation[attributes_to_print]
        for index, row in to_print.iterrows():
            print(formatting_value % tuple(row))
        

if __name__ == '__main__':
    path_ratings = 'ml-100k/u.data'
    delimiter = '\t'
    numRec = 10
    
    #????????? --- TBD: need to replace this section with reading actual live ratings
    newUserID = 377
    ratings_newUser = gLR.new_user_data(path_ratings, delimiter, newUserID)
    file_num = newUserID
    
    #===> testing - Traditional TopN
    full_path_MFmodel = './offline_MFmodel_testing/testingMF_' + str(file_num) + '.npz'
    recommendation_topn = recommenderRSSA.topn_items(full_path_MFmodel, ratings_newUser, newUserID, numRec)
    list_name = "Traditional TopN"
    attributes_to_print = ['item', 'prediction']
    printRSSAlist.print_to_screen(recommendation_topn, newUserID, list_name, attributes_to_print, "%20.0f%20.2f")

    
    #===> testing - Things we think you will hate
    offline_item_ave_full_path = "./offline_prediction_testing/items_prediction_ave.npz"
    attri_name = ['item', 'ave']    
    recommendation_hateitems = recommenderRSSA.hate_items(full_path_MFmodel, ratings_newUser, newUserID, 
                                            offline_item_ave_full_path, attri_name, numRec)
    # print(recommendation_hateitems.columns)                                        
    list_name = "Things we think you will hate"
    attributes_to_print = ['item', 'prediction', 'ave', 'diff']
        # diff = ave - prediction, the higher of diff, the farther of the prediction deviate from the average
    printRSSAlist.print_to_screen(recommendation_hateitems, newUserID, list_name, attributes_to_print, "%20.0f%20.2f%20.2f%20.2f")
    ###:::  diff is small since we are user bias in the algorithm, the higher of bias, the lower of the diff
    ## diff is small : good thing, meaning that the indexing in update function is correct compared to using batch.predict() 
    
    
    #===> testing - Things you will be among the first to try
    offline_items_rating_count_full_path = "./offline_prediction_testing/items_rating_count.npz"
    attri_name = ['item', 'count']
    numTopNprediction = 200
    recommendation_hipitems = recommenderRSSA.novel_items(full_path_MFmodel,ratings_newUser, newUserID, 
                                            offline_items_rating_count_full_path, attri_name, numTopNprediction, numRec)
    list_name = "Things you will be among the first to try"
    attributes_to_print = ['item', 'prediction', 'count']
    printRSSAlist.print_to_screen(recommendation_hipitems, newUserID, list_name, attributes_to_print, "%20.0f%20.2f%20.0f")

    
    #===> testing - Things we have no clue about
    # NEED to check results (see if correct, compare to old highStdFull)
    full_path_MFmodel = './offline_MFmodel_testing/testingMF_' + str(file_num) + '.npz'
    path_resampled_model = './offline_resampled_model_testing/'
    recommendation_highStd = recommenderRSSA.high_std_items(full_path_MFmodel, path_resampled_model, ratings_newUser, newUserID, numRec)
    list_name = "Things we have no clue about: "
    attributes_to_print = ['item', 'prediction', 'std']
    printRSSAlist.print_to_screen(recommendation_highStd, newUserID, list_name, attributes_to_print, "%20.0f%20.2f%20.2f")
    ###  ??? 结果不太对啊， 不同User的recommdations基本相同，只是顺序有差别
    
    
    #===> testing - Things that are controversial
    full_path_MFmodel = './offline_MFmodel_testing/testingMF_' + str(file_num) + '.npz'
    full_path_offline_prediction = './offline_prediction_testing/prediction_all_UI_pair.npz'
    distance_choice = 'eculidean'
    num_neighbors = 20
    recommendation_controversial = recommenderRSSA.controversial_items(full_path_MFmodel, full_path_offline_prediction, 
                                                        ratings_newUser, newUserID, distance_choice, num_neighbors)
    list_name = "Things that are controversial: "
    attributes_to_print = ['item', 'prediction', 'var']
    printRSSAlist.print_to_screen(recommendation_controversial, newUserID, list_name, attributes_to_print, "%20.0f%20.2f%20.2f")
    
    
    
    
    
    
    
    
    
    
    