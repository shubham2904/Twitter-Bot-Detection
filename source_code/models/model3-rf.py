import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def plotROC(rf, xtrain, xtest, ytrain, ytest):
	sns.set_style("whitegrid", {'axes.grid' : False})

	scoresTrain = rf.predict_proba(xtrain)
	scoresTest = rf.predict_proba(xtest)

	y_scoresTrain = []
	y_scoresTest = []
	
	for sc in scoresTrain:
		y_scoresTrain.append(sc[1])
	for sc in scoresTest:
		y_scoresTest.append(sc[1])

	    
	fpr_rf_train, tpr_rf_train, _ = roc_curve(ytrain, y_scoresTrain, pos_label=1)
	fpr_rf_test, tpr_rf_test, _ = roc_curve(ytest, y_scoresTest, pos_label=1)

	np.save('models/model_data/fpr_rf_train', fpr_rf_train)
	np.save('models/model_data/tpr_rf_train', tpr_rf_train)
	np.save('models/model_data/fpr_rf_test', fpr_rf_test)
	np.save('models/model_data/tpr_rf_test', tpr_rf_test)


	plt.plot(fpr_rf_train, tpr_rf_train, color = 'darkblue', label = 'Train AUC: %5f' %auc(fpr_rf_train, tpr_rf_train))
	plt.plot(fpr_rf_test, tpr_rf_test, color = 'red', ls = '--', label = 'Test AUC: %5f' %auc(fpr_rf_test, tpr_rf_test))
	plt.legend(loc = 'lower right')
	plt.title("Random ForestROC Curve")
	plt.xlabel("(FPR) False Positive Rate")
	plt.ylabel("(TPR) True Positive Rate")
	plt.show()

def main():
	
	data = pd.read_pickle("custom_data/processed_dataset.pkl")
	features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified', 'followers_count', 'friends_count', 'statuses_count', 'listed_count_binary', 'bot']
	X = data[features].iloc[:, :-1]
	y = data[features].iloc[:, -1]

	rf = RandomForestClassifier(criterion = 'entropy', min_samples_leaf = 100, min_samples_split = 20)

	xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 101)

	rf = rf.fit(xtrain, ytrain)
	y_pred_train = rf.predict(xtrain)
	y_pred_test = rf.predict(xtest)

	print("Training Accuracy: %.5f" %accuracy_score(ytrain, y_pred_train))
	print("Test Accuracy: %.5f" %accuracy_score(ytest, y_pred_test))

	plotROC(rf, xtrain, xtest, ytrain, ytest)

if __name__ == '__main__':
	main()