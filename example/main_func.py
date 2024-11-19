#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 20:42:15 2022

@author: alphago
"""
import scipy.io as sio
import time
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

training_data = sio.loadmat('training_data.mat')
train_data, train_label = training_data['train_data_set'], training_data['train_label_set']

testing_data = sio.loadmat('testing_data.mat')
test_data, test_label = testing_data['test_data_set'], testing_data['test_label_set']

model = svm.SVC(C = 1, kernel = 'linear', gamma = 10, decision_function_shape = 'ovo')

start = time.time()
model.fit(train_data, train_label.ravel())
end = time.time()

y_pred = model.predict(test_data)


print('Time spent on training the model: ',end-start)

'''
train_score = model.score(train_data, train_label)
print('train_score:', train_score)
test_score = model.score(test_data, test_label)
print('test_score:', test_score)

confusion = confusion_matrix(test_label, y_pred, normalize = 'all')
print(confusion)
'''

count = 0
pop = 0
for i in range(len(y_pred)):
    if y_pred[i] == test_label[i][0]:
        count += 1
    pop += 1
print('Accuracy of the trained model: ',count / pop)
