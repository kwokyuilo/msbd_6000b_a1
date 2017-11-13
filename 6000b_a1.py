import pandas as pd
import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.model_selection import cross_val_score
#features = pd.read_csv('traindata.csv')
#labels = pd.read_csv('trainlabel.csv')
#print(features)

features = genfromtxt('traindata.csv', delimiter=',')
labels = genfromtxt('trainlabel.csv', delimiter=',')
testdata = genfromtxt('testdata.csv', delimiter=',')
#print(features)
features = features/features.max(axis=0)
testdata = testdata/testdata.max(axis=0)
print("Train label %d" % (np.sum(labels)))
#classifier = svm.SVC(kernel='sigmoid', degree=4, C=1000, coef0=-10.0, gamma=10)
#classifier = svm.SVC(kernel='rbf', degree=4, C=100, coef0=-10.0, gamma=1)
classifier = svm.SVC(kernel='poly', degree=4, C=1, coef0=10.0, gamma=0.1)
labels = np.array(labels).flatten()

print("Number of features %d, Total numnber of data: %d" % (features.shape[1], features.shape[0]))
scores = cross_val_score(classifier, features, labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
classifier.fit(features,labels)

np.savetxt('result.csv', classifier.predict(testdata), delimiter=',')
print(np.sum(classifier.predict(testdata)))


