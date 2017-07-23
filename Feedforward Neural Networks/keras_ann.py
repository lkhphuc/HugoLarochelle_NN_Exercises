# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import mlpython.datasets.store as dataset_store

# Importing the dataset
trainset,validset,testset = dataset_store.get_classification_problem('ocr_letters')
# Get dataset into Numpy array
X_train = np.asarray([trainset.data.mem_data[0][0]])
y_train = trainset.data.mem_data[1][0]
for x,y in trainset:
    X_train = np.vstack([X_train, x])
    y_train = np.vstack([y_train, y])
#remove the first redundant example
X_train = X_train[1:,:]
y_train = y_train[1:,:]

X_val = validset.data.mem_data[0][0]
y_val = validset.data.mem_data[1][0]
for x,y in validset:
    X_val = np.vstack([X_val, x])
    y_val = np.vstack([y_val, y])
X_val = X_val[1:,:]
y_val = y_val[1:,:]

# Turn y_val into one hot encoder
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(sparse=False)
y_train = onehot.fit_transform(y_train)
y_val = onehot.fit_transform(y_val)

# Import ANN libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# Fitting classifier to the Training set
classifier = Sequential()

# Add first layer
classifier.add(Dense(units=64, kernel_initializer='uniform', activation='sigmoid', input_dim= X_train.shape[1]))
# Add second layer
classifier.add(Dense(units=32, kernel_initializer='uniform', activation='sigmoid'))

# Add output layer
classifier.add(Dense(units=26, kernel_initializer='uniform', activation='softmax'))

# Compile classifier
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the classifier
classifier.fit(X_train, y_train, batch_size=32, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_val)
y_pred = np.argmax(y_pred, axis=1)[:, None]
for i in range(y_pred.shape[0]):
    y_pred[i] = 1 * (y_pred[i] == np.amax(y_pred[i]))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred)

accu = 0.0
for i in range(y_pred.shape[0]):
    accu += cm[i, i]
accu = accu / y_pred.shape[0]
