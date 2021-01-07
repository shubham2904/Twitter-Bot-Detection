import os

def main():
	'''
	sequence:
	1. run analysis and feature-extraction
	2. run all models
	3. model comparision
	'''
	print("\n******************************************\n")
	print("         Running Data Analysis Module")
	print("\n******************************************\n")
	os.system("python data_analysis.py")
	
	print("\n******************************************\n")
	print("      Running Feature Extraction Module")
	print("\n******************************************\n")
	os.system("python feature_extraction_and_engg.py")
	
	print("\n******************************************\n")
	print("    Running Multinomial Naive Baye's Model")
	print("\n******************************************\n")
	os.system("python models/model1-mnb.py")
	
	print("\n******************************************\n")
	print("         Running Decision Tree Model")
	print("\n******************************************\n")
	os.system("python models/model2-dt.py")
	
	print("\n******************************************\n")
	print("         Running Randon Forest Model")
	print("\n******************************************\n")
	os.system("python models/model3-rf.py")
	
	print("\n******************************************\n")
	print("        Running Our Own Made Module")
	print("\n******************************************\n")
	os.system("python models/model4_own.py")
	
	print("\n******************************************\n")
	print("     Running a comparision of the models")
	print("\n******************************************\n")
	os.system("python models/model_comparison.py")


if __name__ == '__main__':
	main()