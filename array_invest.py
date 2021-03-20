import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # DIsable all the TensorFlow Extra Debug Notification.

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

import keras.utils

import innvestigate
import innvestigate.utils as iutils


def get_data(train_data_path, val_data_path):
    # get train data
    train = pd.read_csv(train_data_path)

    # get val data
    val = pd.read_csv(val_data_path)

    return train, val

def get_combined_data(train_data_path, val_data_path):

    # train and validation dataset are imported
    train, val = get_data(train_data_path, val_data_path)

    # The Ground truth of both train and validation dataset are dropped
    # and stored in the variables trainGT and valGT

    trainGT = train.Class
    train.drop(['Class'], axis=1, inplace=True)

    valGT = val.Class
    val.drop(['Class'], axis=1, inplace=True)

    # data vector for train and validation  are scaled between 0 and 1
    # and structured as Pandas Data Frame

    train = train.values  # returns a numpy array
    min_max_scaler_train = preprocessing.MinMaxScaler()
    train = min_max_scaler_train.fit_transform(train)
    train = pd.DataFrame(train)

    val = val.values  # returns a numpy array
    min_max_scaler_test = preprocessing.MinMaxScaler()
    val = min_max_scaler_test.fit_transform(val)
    val = pd.DataFrame(val)

    return train, trainGT, val, valGT



# The dataset (both val and test) are preprocessed and
train, trainGT, val, valGT = get_combined_data('mytrain.csv','myval.csv')


model = keras.models.Sequential([
    keras.layers.Dense(128, kernel_initializer='normal',input_dim = train.shape[1], activation='relu', name="dense_zero"),
    #keras.layers.Dense(256, kernel_initializer='normal',activation='relu', name="dense_one"),
    keras.layers.Dense(64, activation="relu", name="dense_two"),
    keras.layers.Dense(2, activation="softmax", name="dense_three"),
])

#model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(train,
          keras.utils.to_categorical(trainGT), #the ground truth vector is prepared as categorical
          epochs=15,
          batch_size=128
          )

model.save("model_cc")


## Testing and visualization ##

#loading the model
model = keras.models.load_model("model_cc")
# Stripping the softmax activation from the model
model_wo_sm = iutils.keras.graph.model_wo_softmax(model)
# Creating an analyzer
gradient_analyzer = innvestigate.analyzer.Gradient(model_wo_sm)


f = open("exai_out.txt", "w") # file collecting the analyzer output

count = 0
vector_sum = np.zeros(len(val.loc[[0]])) # vector collecting the contribution of all the analyzer outputs
for i in range(0, val.shape[0]):
    count += 1
    # Applying the analyzer
    analysis = gradient_analyzer.analyze(val.loc[[i]])
    analysis = analysis[0]
    vector_sum = np.add(vector_sum, analysis)
    strToPrint = str(analysis.tolist())
    f.write(strToPrint + "\n")

f.close()

vector_mean = np.true_divide(vector_sum, count) # the contribution of all the analyzer outputs are averaged
print(vector_mean)

plt.plot(vector_mean)
plt.show()

