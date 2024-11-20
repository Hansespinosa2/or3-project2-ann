import numpy as np

def load_data(folder_path:str)->tuple:
	'''
		Input the folder path that holds the data sets. The folder must have appropriately named files.
	'''
	X_train = np.loadtxt(f'{folder_path}/train_data_set.csv', delimiter=',', skiprows=1)
	X_test = np.loadtxt(f'{folder_path}/test_data_set.csv', delimiter=',', skiprows=1)
	y_train = np.loadtxt(f'{folder_path}/train_label_set.csv', delimiter=',', skiprows=1).ravel()
	y_test = np.loadtxt(f'{folder_path}/test_label_set.csv', delimiter=',', skiprows=1).ravel()
	return X_train, y_train, X_test, y_test