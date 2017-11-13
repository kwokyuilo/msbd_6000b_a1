import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

import numpy as np

features = np.loadtxt('traindata.csv', delimiter=',', dtype=np.float32)
labels = np.loadtxt('trainlabel.csv', delimiter=',', dtype=np.float32)
test = np.loadtxt('testdata.csv', delimiter=',', dtype=np.float32)


print("Number of features:%d"%(features.shape[1]))
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(256, input_dim=features.shape[1],activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(features, labels, epochs=100, batch_size=10)

scores = model.evaluate(features,labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predict = model.predict(test)
predict = [round(x[0]) for x in predict]
np.savetxt('result_nn.csv', predict, delimiter=',')