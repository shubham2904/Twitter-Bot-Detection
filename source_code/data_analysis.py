import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
import seaborn as sns


def NaNHeatmap(df):
	# Heatmap of all Missing values
	plt.figure(figsize = (10, 6))
	sns.heatmap(df.isnull(), cmap = 'viridis', yticklabels = False, cbar = False)
	plt.tight_layout()
	plt.show()

def FriendsVsFollowersRelation(bots, nonbots):
	plt.figure(figsize=(10, 5))
	plt.subplot(2, 1, 1)
	plt.title('Friends vs Followers For Bots')
	sns.regplot(bots.friends_count, bots.followers_count, label = 'Bots', color = 'green')
	plt.ylim(0, 100)
	plt.xlim(0, 100)
	plt.tight_layout()

	plt.subplot(2, 1, 2)
	plt.title('Friends vs Followers For Non Bots')
	sns.regplot(nonbots.friends_count, nonbots.followers_count, label = 'NonBots', color = 'blue')
	plt.ylim(0, 100)
	plt.xlim(0, 100)
	plt.tight_layout()

	plt.show()

def ListedCountlessthan5(bots, nonbots):
	plt.figure(figsize = (10, 5))
	plt.plot(bots.listed_count, color = 'red', label = 'Bots')
	plt.plot(nonbots.listed_count, color = 'blue', label = 'NonBots')
	plt.legend(loc = 'upper left')
	plt.ylim(10000, 20000)
	plt.show()

def ListedCountThreshold(bots, nonbots, threshold):
	bots_listed_count_df = bots[bots.listed_count < threshold]
	nonbots_listed_count_df = nonbots[nonbots.listed_count < threshold]

	bots_verified_df = bots_listed_count_df[bots_listed_count_df.verified == False]
	bots_screenname_has_bot_df_ = bots_verified_df[(bots_verified_df.screen_name.str.contains("bot", case = False) == True)].shape

	plt.figure(figsize = (12, 7))

	plt.subplot(2, 1, 1)
	plt.plot(bots_listed_count_df.friends_count, color = 'red', label = 'Bots Friends')
	plt.plot(nonbots_listed_count_df.friends_count, color = 'blue', label = 'NonBots Friends')
	plt.legend(loc = 'upper left')

	plt.subplot(2, 1, 2)
	plt.plot(bots_listed_count_df.followers_count, color = 'red', label = 'Bots Followers')
	plt.plot(nonbots_listed_count_df.followers_count, color = 'blue', label = 'NonBots Followers')
	plt.legend(loc = 'upper left')
	plt.show()

def SpearmanCorrelation(df):
	print(df.corr(method = 'spearman'))
	plt.figure(figsize = (8, 4))
	sns.heatmap(df.corr(method = 'spearman'), cmap = 'coolwarm', annot = True)
	plt.tight_layout()
	plt.show()


def main():
	## Loading Dataset ##

	outerpath = 'kaggle_data/'
	path = outerpath + 'training_data_2_csv_UTF.csv'

	training_data = pd.read_csv(path)
	bots = training_data[training_data.bot == 1]
	nonbots = training_data[training_data.bot == 0]

	print("\n\n## 1. Identifying Missingness of the data ##")
	NaNHeatmap(training_data)

	print("\n## 2. Analysing Listed Count Feature ##")
	ListedCountlessthan5(bots, nonbots)
	print("> Bots with listed_count less than 5: {0}".format(bots[(bots.listed_count < 5)].shape[0]))

	ListedCountThreshold(bots, nonbots, 16000)


	print("\n\n## 3. Looking at the relation between friends vs followers in bots and nonbots ##")
	print("> A few values of friends_count/followers_count: ")
	print((bots.friends_count / bots.followers_count)[:10])
	FriendsVsFollowersRelation(bots, nonbots)

	print("\n\nThe ratio of the two features:")
	bots['friends_by_followers'] = bots.friends_count/bots.followers_count
	print('> Ratio less than 1 in bots: {0}'.format(bots[bots.friends_by_followers < 1].shape[0]))

	nonbots['friends_by_followers'] = nonbots.friends_count/nonbots.followers_count
	print('> Ratio less than 1 in nonbots: {0}'.format(nonbots[nonbots.friends_by_followers < 1].shape[0]))
	


	print("\n\n## 4. Basic characterstics of bots ##")
	condition = (bots.screen_name.str.contains("bot", case = False) == True) | (bots.description.str.contains("bot", case = False) == True) | (bots.verified == False) | (bots.location.isnull())
	bots['screen_name_binary'] = (bots.screen_name.str.contains("bot", case = False) == True)
	bots['location_binary'] = (bots.location.isnull())
	bots['verified_binary'] = (bots.verified == False)
	print("> Condition satisfied in bots dataset: %s" %bots.shape[0])

	condition = (nonbots.screen_name.str.contains("bot", case = False) == False) | (nonbots.description.str.contains("bot", case=False) == False) | (nonbots.verified == True) | (nonbots.location.isnull() == False)
	nonbots['screen_name_binary'] = (nonbots.screen_name.str.contains("bot", case = False) == False)
	nonbots['location_binary'] = (nonbots.location.isnull() == False)
	nonbots['verified_binary'] = (nonbots.verified == True)
	
	print("> Condition satisfied in bots dataset: %s\n" %nonbots.shape[0])

	df = pd.concat([bots, nonbots])
	df.to_pickle("custom_data/primarydataset.pkl")

	## 5. Spearman correlation ##
	SpearmanCorrelation(df)

if __name__ == '__main__':
	main()
