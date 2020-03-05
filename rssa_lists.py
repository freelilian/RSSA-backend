'''
	Recommending RSSA lists for live user
    All offline models and dataset are ready in './data_dummy'
'''
from dataset_split import datasetSplit
from API import recommenderRSSA, printRSSAlist

if __name__ == '__main__':
   
    #===> import live user ratings from testset
    fullpath_test = './data_dummy/test.npz'
    attri_name = ['user', 'item', 'rating', 'timestamp']
    [ratings_liveUser, liveUserID] = datasetSplit.load_live_user_ratings(fullpath_test, attri_name)
    
    #===> address nessary variables and offline model pathes/fullpathes
    numRec = 10
    fullpath_MFmodel = './data_dummy/MFmodel.npz'
    fullpath_full_prediction = './data_dummy/prediction_full_UI_pair.npz'
    fullpath_items_prediction_ave = './data_dummy/items_prediction_ave.npz'
    fullpath_items_rating_count = './data_dummy/items_rating_count.npz'
    path_resampled_models = './data_dummy/resampled_models/'
    
    #===> Traditional TopN
    recommendation_topn = recommenderRSSA.topn_items(fullpath_MFmodel, ratings_liveUser, liveUserID, numRec)
    list_name = "Traditional TopN"
    attributes_to_print = ['item', 'prediction']
    printRSSAlist.print_to_screen(recommendation_topn, liveUserID, list_name, attributes_to_print, "%20.0f%20.2f")
    
    #===> Things we think you will hate
    attri_name = ['item', 'ave']
    recommendation_hateitems = recommenderRSSA.hate_items(fullpath_MFmodel, ratings_liveUser, liveUserID, 
                                            fullpath_items_prediction_ave, attri_name, numRec)
    list_name = "Things we think you will hate"
    attributes_to_print = ['item', 'prediction', 'ave', 'diff']
    printRSSAlist.print_to_screen(recommendation_hateitems, liveUserID, list_name, attributes_to_print, "%20.0f%20.2f%20.2f%20.2f")

    #===> Things you will be among the first to try
    attri_name = ['item', 'count']
    numTopNprediction = 200
    recommendation_hipitems = recommenderRSSA.novel_items(fullpath_MFmodel, ratings_liveUser, liveUserID, 
                                            fullpath_items_rating_count, attri_name, numTopNprediction, numRec)
    list_name = "Things you will be among the first to try"
    attributes_to_print = ['item', 'prediction', 'count']
    printRSSAlist.print_to_screen(recommendation_hipitems, liveUserID, list_name, attributes_to_print, "%20.0f%20.2f%20.0f")
    
    #===> Things we have no clue about
    recommendation_highStd = recommenderRSSA.high_std_items(fullpath_MFmodel, path_resampled_models, 
                                                                ratings_liveUser, liveUserID, numRec)
    list_name = "Things we have no clue about: "
    attributes_to_print = ['item', 'prediction', 'std']
    printRSSAlist.print_to_screen(recommendation_highStd, liveUserID, list_name, attributes_to_print, "%20.0f%20.2f%20.2f")

    #===> Things that are controversial
    distance_choice = 'eculidean'
    num_neighbors = 20
    recommendation_controversial = recommenderRSSA.controversial_items(fullpath_MFmodel, fullpath_full_prediction, 
                                                        ratings_liveUser, liveUserID, distance_choice, num_neighbors)
    list_name = "Things that are controversial: "
    attributes_to_print = ['item', 'prediction', 'var']
    printRSSAlist.print_to_screen(recommendation_controversial, liveUserID, list_name, attributes_to_print, "%20.0f%20.2f%20.2f")
    