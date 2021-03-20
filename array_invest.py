'''
The code trains a NN accepting as input features vectors (exploiting a training dataset)
Then the code applies the explainable AI methods to the produced NN model in order to generate a predictor
including the evaluation of features influence on the choice of the estimated class and produces report and plot
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

import keras.utils

import innvestigate
import innvestigate.utils as iutils

# Get the csv files and load them as pandas data frame
def get_data(train_data_path, val_data_path):
    # Get train data
    train = pd.read_csv(train_data_path)

    # Get validation data
    val = pd.read_csv(val_data_path)

    return train, val

# The pandas data frame for train and validation are splitted in "data" and "ground truth"
# moreover the csv head is removed. The data are scaled between 0 and 1.
def get_combined_data(train_data_path, val_data_path):

    # Train and validation dataset are imported
    train, val = get_data(train_data_path, val_data_path)

    # The Ground truth of both train and validation dataset are dropped
    # and stored in the variables trainGT and valGT

    trainGT = train.Class
    train.drop(['Class'], axis=1, inplace=True)

    valGT = val.Class
    val.drop(['Class'], axis=1, inplace=True)

    # Data vector for train and validation are scaled between 0 and 1
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



# The dataset (both validation and test) are loaded and preprocessed
train, trainGT, val, valGT = get_combined_data('mytrain.csv','myval.csv')

# The NN is created
model = keras.models.Sequential([
    keras.layers.Dense(128, kernel_initializer='normal',input_dim = train.shape[1], activation='relu', name="dense_zero"),
    #keras.layers.Dense(256, kernel_initializer='normal',activation='relu', name="dense_one"),
    keras.layers.Dense(64, activation="relu", name="dense_two"),
    keras.layers.Dense(2, activation="softmax", name="dense_three"),
])


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# The training process is executed
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

# The validation dataset is processed with the modified model "gradient_analyzer"
# The output vectors, describing the features influence and the final prediction, are store in the file
# and averaged in a resuming vector

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

# The contribution of all the analyzer outputs are averaged and plotted
vector_mean = np.true_divide(vector_sum, count)
print(vector_mean)

plt.plot(vector_mean)
plt.show()

