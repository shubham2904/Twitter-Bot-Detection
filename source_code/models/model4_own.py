import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
import seaborn as sns
import sklearn.metrics as metrics

def GetBagOfWords():
    bow = pd.read_csv("custom_data/BotBagofWords.csv").columns.tolist()
    bag = r''
    for word in bow:
        bag = bag + word + '|'
    bag = bag[:-1]
    
    return bag

class twitter_bot(object):
    def __init__(self):
        pass

    def perform_train_test_split(df):
        msk = np.random.rand(len(df)) < 0.75
        train, test = df[msk], df[~msk]
        xtrain, ytrain = train, train.iloc[:,-1]
        xtest, ytest = test, test.iloc[:, -1]
        return (xtrain, ytrain, xtest, ytest)

    def bot_prediction_algorithm(df):
        # creating copy of dataframe
        traindf = df.copy()
        # performing feature engineering on id and verfied columns
        # converting id to int
        traindf['id'] = traindf.id.apply(lambda x: int(x))
        #traindf['friends_count'] = traindf.friends_count.apply(lambda x: int(x))
        traindf['followers_count'] = traindf.followers_count.apply(lambda x: 0 if x == 'None' else int(x))
        traindf['friends_count'] = traindf.friends_count.apply(lambda x: 0 if x == 'None' else int(x))
        #We created two bag of words because more bow is excessive on test data, so on all small dataset we check less
        if traindf.shape[0] > 600:
            #bag_of_words_for_bot
            bag_of_words_bot = GetBagOfWords()
        else:
            # bag_of_words_for_bot
            bag_of_words_bot = r'bot|b0t|cannabis|mishear|updates every'

        # converting verified into vectors
        traindf['verified'] = traindf.verified.apply(lambda x : 1 if ((x == True) or x == 'TRUE') else 0)

        # check if the name contains bot or screenname contains b0t
        condition = ((traindf.name.str.contains(bag_of_words_bot, case = False, na = False)) |
                     (traindf.description.str.contains(bag_of_words_bot, case = False, na = False)) |
                     (traindf.screen_name.str.contains(bag_of_words_bot, case = False, na = False)) |
                     (traindf.status.str.contains(bag_of_words_bot, case = False, na = False)))
        predicteddf = traindf[condition]  # these all are bots
        predicteddf.bot = 1
        predicteddf = predicteddf[['id', 'bot']]

        # check if the user is verified
        verifieddf = traindf[~condition]
        condition = (verifieddf.verified == 1)  # these all are nonbots
        predicteddf1 = verifieddf[condition][['id', 'bot']]
        predicteddf1.bot = 0
        predicteddf = pd.concat([predicteddf, predicteddf1])

        # check if description contains buzzfeed
        buzzfeed_df = verifieddf[~condition]
        condition = (buzzfeed_df.description.str.contains("buzzfeed", case = False, na = False))  # these all are nonbots
        predicteddf1 = buzzfeed_df[buzzfeed_df.description.str.contains("buzzfeed", case = False, na = False)][['id', 'bot']]
        predicteddf1.bot = 0
        predicteddf = pd.concat([predicteddf, predicteddf1])

        # check if listed_count>16000
        listed_countdf = buzzfeed_df[~condition]
        listed_countdf.listed_count = listed_countdf.listed_count.apply(lambda x: 0 if x == 'None' else x)
        listed_countdf.listed_count = listed_countdf.listed_count.apply(lambda x: int(x))
        condition = (listed_countdf.listed_count > 16000)  # these all are nonbots
        predicteddf1 = listed_countdf[condition][['id', 'bot']]
        predicteddf1.bot = 0
        predicteddf = pd.concat([predicteddf, predicteddf1])

        #remaining
        predicteddf1 = listed_countdf[~condition][['id', 'bot']]
        predicteddf1.bot = 0 # these all are nonbots
        predicteddf = pd.concat([predicteddf, predicteddf1])
        return predicteddf

    def get_predicted_and_true_values(features, target):
        y_pred, y_true = twitter_bot.bot_prediction_algorithm(features).bot.tolist(), target.tolist()
        return (y_pred, y_true)

    def get_accuracy_score(df):
        (xtrain, ytrain, xtest, ytest) = twitter_bot.perform_train_test_split(df)
        # predictions on training data
        y_pred_train, y_true_train = twitter_bot.get_predicted_and_true_values(xtrain, ytrain)
        train_acc = metrics.accuracy_score(y_pred_train, y_true_train)
        #predictions on test data
        y_pred_test, y_true_test = twitter_bot.get_predicted_and_true_values(xtest, ytest)
        test_acc = metrics.accuracy_score(y_pred_test, y_true_test)
        return (train_acc, test_acc)

    def plot_roc_curve(df):
        (xtrain, ytrain, xtest, ytest) = twitter_bot.perform_train_test_split(df)
        # Train ROC
        y_pred_train, y_true = twitter_bot.get_predicted_and_true_values(xtrain, ytrain)
        scores = np.linspace(start = 0.01, stop = 0.9, num = len(y_true))
        fpr_train, tpr_train, threshold = metrics.roc_curve(y_pred_train, scores, pos_label = 0)
        plt.plot(fpr_train, tpr_train, label = 'Train AUC: %5f' % metrics.auc(fpr_train, tpr_train), color = 'darkblue')
        #Test ROC
        y_pred_test, y_true = twitter_bot.get_predicted_and_true_values(xtest, ytest)
        scores = np.linspace(start = 0.01, stop = 0.9, num = len(y_true))
        fpr_test, tpr_test, threshold = metrics.roc_curve(y_pred_test, scores, pos_label = 0)
        plt.plot(fpr_test,tpr_test, label = 'Test AUC: %5f' %metrics.auc(fpr_test, tpr_test), ls = '--', color = 'red')
        #Misc
        plt.xlim([-0.1,1])
        plt.title("Reciever Operating Characteristic (ROC)")
        plt.legend(loc = 'lower right')
        plt.xlabel("(FPR) False Positive Rate")
        plt.ylabel("(TPR) True Positive Rate")
        plt.show()


if __name__ == '__main__':
    #start = time.time()
    outerpath = 'kaggle_data/'
    traindf = pd.read_csv(outerpath + 'training_data_2_csv_UTF.csv')
    testdf = pd.read_csv(outerpath + 'test_data_4_students.csv', sep = '\t', encoding='ISO-8859-1')
    print("Train Accuracy: %.5f" %twitter_bot.get_accuracy_score(traindf)[0])
    print("Test Accuracy: %.5f" %twitter_bot.get_accuracy_score(traindf)[1])

    #predicting test data results
    predicteddf = twitter_bot.bot_prediction_algorithm(testdf)
    #plotting the ROC curve
    twitter_bot.plot_roc_curve(traindf)