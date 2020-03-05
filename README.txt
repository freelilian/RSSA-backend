*------------------------------------------------------------------------------------------------------------------
Algorithms for lively recommending RSSA lists V4
*------------------------------------------------------------------------------------------------------------------
	PS: using ml-25m, lenskit 0.3.0
*------------------------------------------------------------------------------------------------------------------
	Updated list:
		- test on ml-latest-small, works!
		- test on ml-25m dataset, run out of memory when generating full user-item pairs
			- Segmentation fault (core dumped)
		- make sure each user rated >= 20 movies in trainset
		
*------------------------------------------------------------------------------------------------------------------
	wrap_offlineModel.py
		Compute some offline models and save them: 
			- offline_MFmodels
			- full prediction
			- rating counts, for hip items
			- average ratings, For hate items
			
		Containing:
			- class offlineMFModel
				- def UIfeatures
				- def save_model
				- def load_model
			- class offlinePrediction
				- def full_offline_UIpairs
				- def save_dataset
				- def load_dataset
				- def predictor
				- def merge_prediction_rating
			- class offlineCalculation
				- def item_calculate_ave_prediction
				- def item_count_ratings
				- def singular_value_weight (unused)
			- main: test the functionality of the defined classes and functions
			
	resampleOffline.py
		Compute and save the resampled 20 MFmodels
		Containing:
			- def train_offline_resample_models
			- main: test the functionality of the defined function
			
	generateLiveRating.py
		Get one user's ratings as the newUser's live rating data from the offline dataset and treat the remaining as the training dataset
		Containing:
			- def split_offline_dataset
			- def new_user_data
			- main: test the functionality of the defined functions
			
	wrap_update.py
		Compute live data to serve the live functions of the 5 RSSA lists, also compute RMSE to evaluate the accuracy of the live function:
		Containing
			- class liveFunction
				- def newUser_rating_noBias
				- def newUser_predicting
				- def computeRMSE
				- def relative_ave
				- def similarity_user_features
			- main: test the functionality of the defined functions and print RMSE
	
	topnLive.py
		Containing:
			- def descending_prediction
			- main: test the functionality of list -- Traditional TopN
			
	hateitemsLive.py
		Containing:
			- def descending_distance_from_ave
			- main: test the functionality of list -- Things we think you will hate
	
	hipitemsLive.py
		Containing:
			- def novel
			- main: test the functionality of list -- Things you will be among the first to try
	highStdLive.py
		Containing:
			- def load_resampled_model_path
			- def live_user_prediction_resampling
			- def concatenate_prediction
			- def check_items (not really used maybe)
			- def descending_std
			- main: test the functionality of list -- Things we have no clue about
	controversialLive.py
		Containing:
			- def find_neighbors
			- def extract_neighbor_predictions
			- def variance
			- main: test the functionality of list -- Things that are controversial
	
	API.py
		APIs for calling RSSA recommender with an live newUser's data and print all the five recommendation lists
		Containing:
			- class recommenderRSSA
				- def topn_items
				- def hate_items
				- def novel_items
				- def high_std_items
				- def controversial_items
			- class printRSSAlist
				- def print_to_screen
			- main: test the functionality of the APIs
				
	dataset_split.py
		Split offline dataset to trainset and testset, use testset to generate liveUser, then to test the update functions
		Containing:
			- class datasetSplit
				- def split_dataset
				- def load_data
				- def load_live_user_ratings
			- main: test the functionality of the defined functions
	
	train_offline_model.py
		Train all nessary offline models:
			- MFmodel
			- 20 Resampled models
		Pre-calculate offline dataset:
			- prediction_full_UI_pair
			- items_prediction_ave
			- items_rating_count
		Save in ./data_dummy
	rssa_lists.py
		Test all RSSA lists
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	