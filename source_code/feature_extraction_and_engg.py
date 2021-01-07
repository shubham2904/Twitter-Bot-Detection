import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
import seaborn as sns

def GetBagOfWords():
	bow = pd.read_csv("custom_data/BotBagofWords.csv").columns.tolist()
	bag = r''
	for word in bow:
		bag = bag + word + '|'
	bag = bag[:-1]
	
	return bag

def main():

	## Loading Dataset ##
	outerpath = 'kaggle_data/'
	path = outerpath + 'training_data_2_csv_UTF.csv'
	train = pd.read_csv(path)

	# Feature Engineering
	bag = GetBagOfWords()
	train['screen_name_binary'] = train.screen_name.str.contains(bag, na = False, case = False)
	train['name_binary'] = train.name.str.contains(bag, na = False, case = False)
	train['description_binary'] = train.description.str.contains(bag, na = False, case = False)
	train['status_binary'] = train.status.str.contains(bag, na = False, case = False)

	# Feature Extraction
	train['listed_count_binary'] = ((train.listed_count > 20000) == False)
	features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified', 'followers_count', 'friends_count', 'statuses_count', 'listed_count_binary', 'bot']

	#train.to_csv("custom_data/Model_Data.csv")
	train.to_pickle("custom_data/processed_dataset.pkl")


if __name__ == '__main__':
	main()