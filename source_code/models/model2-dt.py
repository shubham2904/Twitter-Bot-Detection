import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split

def PlotROC(dt, xtrain, xtest, ytrain, ytest):
	sns.set(font_scale = 1.5)
	sns.set_style("whitegrid", {'axes.grid' : False})

	scoresTrain = dt.predict_proba(xtrain)
	scoresTest = dt.predict_proba(xtest)

	y_scoresTrain = []
	y_scoresTest = []

	for sc in scoresTrain:
		y_scoresTrain.append(sc[1])
	for sc in scoresTest:
		y_scoresTest.append(sc[1])

	    
	fpr_dt_train, tpr_dt_train, _ = roc_curve(ytrain, y_scoresTrain, pos_label = 1)
	fpr_dt_test, tpr_dt_test, _ = roc_curve(ytest, y_scoresTest, pos_label = 1)

	np.save('models/model_data/fpr_dt_train', fpr_dt_train)
	np.save('models/model_data/tpr_dt_train', tpr_dt_train)
	np.save('models/model_data/fpr_dt_test', fpr_dt_test)
	np.save('models/model_data/tpr_dt_test', tpr_dt_test)

	plt.plot(fpr_dt_train, tpr_dt_train, color = 'darkblue', label = 'Train AUC: %5f' %auc(fpr_dt_train, tpr_dt_train))
	plt.plot(fpr_dt_test, tpr_dt_test, color = 'red', ls='--', label = 'Test AUC: %5f' %auc(fpr_dt_test, tpr_dt_test))
	plt.legend(loc = 'lower right')
	plt.title("Decision Tree ROC Curve")
	plt.xlabel("(FPR) False Positive Rate")
	plt.ylabel("(TPR) True Positive Rate")
	plt.show()

def main():

	data = pd.read_pickle("custom_data/processed_dataset.pkl")
	features = ['screen_name_binary', 'name_binary', 'description_binary', 'status_binary', 'verified', 'followers_count', 'friends_count', 'statuses_count', 'listed_count_binary', 'bot']
	X = data[features].iloc[:,:-1]
	y = data[features].iloc[:,-1]

	dt = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 50, min_samples_split = 10)
	xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 101)

	dt = dt.fit(xtrain, ytrain)
	y_pred_train = dt.predict(xtrain)
	y_pred_test = dt.predict(xtest)

	print("Training Accuracy: %.5f" %accuracy_score(ytrain, y_pred_train))
	print("Test Accuracy: %.5f" %accuracy_score(ytest, y_pred_test))

	PlotROC(dt, xtrain, xtest, ytrain, ytest)


if __name__ == '__main__':
	main()
