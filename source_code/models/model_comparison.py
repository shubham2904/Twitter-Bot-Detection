import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split

from model4_own import twitter_bot

def main():
	df = pd.read_pickle("custom_data/primarydataset.pkl")

	plt.figure(figsize = (14, 10))
	(X_train, y_train, X_test, y_test) = twitter_bot.perform_train_test_split(df)
	#Train ROC
	y_pred_train, y_true = twitter_bot.get_predicted_and_true_values(X_train, y_train)
	scores = np.linspace(start = 0, stop = 1, num = len(y_true))
	fpr_botc_train, tpr_botc_train, threshold = metrics.roc_curve(y_pred_train, scores, pos_label = 0)

	#Test ROC
	y_pred_test, y_true = twitter_bot.get_predicted_and_true_values(X_test, y_test)
	scores = np.linspace(start = 0, stop = 1, num = len(y_true))
	fpr_botc_test, tpr_botc_test, threshold = metrics.roc_curve(y_pred_test, scores, pos_label = 0)


	fpr_rf_train = np.load('models/model_data/fpr_rf_train.npy')
	tpr_rf_train = np.load('models/model_data/tpr_rf_train.npy')
	fpr_dt_train = np.load('models/model_data/fpr_dt_train.npy')
	tpr_dt_train = np.load('models/model_data/tpr_dt_train.npy')
	fpr_mnb_train = np.load('models/model_data/fpr_mnb_train.npy')
	tpr_mnb_train = np.load('models/model_data/tpr_mnb_train.npy')

	#Train ROC
	plt.subplot(2, 2, 1)
	plt.plot(fpr_botc_train, tpr_botc_train, label = 'Our Classifier AUC: %5f' % metrics.auc(fpr_botc_train,tpr_botc_train), color='darkblue')
	plt.plot(fpr_rf_train, tpr_rf_train, label = 'Random Forest AUC: %5f' %auc(fpr_rf_train, tpr_rf_train))
	plt.plot(fpr_dt_train, tpr_dt_train, label = 'Decision Tree AUC: %5f' %auc(fpr_dt_train, tpr_dt_train))
	plt.plot(fpr_mnb_train, tpr_mnb_train, label = 'MultinomialNB AUC: %5f' %auc(fpr_mnb_train, tpr_mnb_train))
	plt.title("Training Set ROC Curve")
	plt.legend(loc = 'lower right')
	plt.xlabel("(FPR) False Positive Rate")
	plt.ylabel("(TPR) True Positive Rate")
	

	fpr_rf_test = np.load('models/model_data/fpr_rf_test.npy')
	tpr_rf_test = np.load('models/model_data/tpr_rf_test.npy')
	fpr_dt_test = np.load('models/model_data/fpr_dt_test.npy')
	tpr_dt_test = np.load('models/model_data/tpr_dt_test.npy')
	fpr_mnb_test = np.load('models/model_data/fpr_mnb_test.npy')
	tpr_mnb_test = np.load('models/model_data/tpr_mnb_test.npy')

	#Test ROC
	plt.subplot(2, 2, 2)
	plt.plot(fpr_botc_test,tpr_botc_test, label = 'Our Classifier AUC: %5f' %metrics.auc(fpr_botc_test,tpr_botc_test), color = 'darkblue')
	plt.plot(fpr_rf_test, tpr_rf_test, label = 'Random Forest AUC: %5f' %auc(fpr_rf_test, tpr_rf_test))
	plt.plot(fpr_dt_test, tpr_dt_test, label = 'Decision Tree AUC: %5f' %auc(fpr_dt_test, tpr_dt_test))
	plt.plot(fpr_mnb_test, tpr_mnb_test, label = 'MultinomialNB AUC: %5f' %auc(fpr_mnb_test, tpr_mnb_test))
	plt.title("Test Set ROC Curve")
	plt.legend(loc = 'lower right')
	plt.xlabel("(FPR) False Positive Rate")
	plt.ylabel("(TPR) True Positive Rate")
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()