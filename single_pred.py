import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # DIsable all the TensorFlow Extra Debug Notification.

import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

import keras.utils

import innvestigate
import innvestigate.utils as iutils


def get_data(val_data_path):
    val = pd.read_csv(val_data_path)

    return train, val

def get_combined_data(val_data_path):

    val = pd.read_csv(val_data_path)
    # The Ground truth of both validation dataset is dropped
    # and stored in the variable valGT

    valGT = val.Class
    val.drop(['Class'], axis=1, inplace=True)

    # data vector for train and validation  are scaled between 0 and 1
    # and structured as Pandas Data Frame

    val = val.values  # returns a numpy array
    min_max_scaler_test = preprocessing.MinMaxScaler()
    val = min_max_scaler_test.fit_transform(val)
    val = pd.DataFrame(val)

    return val, valGT

# The dataset (both val and test) are preprocessed and
val, valGT = get_combined_data('baseline2.csv')

if len(sys.argv)==2:
    row_num = int(sys.argv[1])
    if row_num < len(valGT):
        print("I'll try test the row #: ", row_num)
    else:
        print("Row is out of index")
        sys.exit(1)
else:
    print("Wrong number of arguments")
    sys.exit(1)

model = keras.models.load_model("model_cc")
# Stripping the softmax activation from the model
model_wo_sm = iutils.keras.graph.model_wo_softmax(model)
# Creating an analyzer
gradient_analyzer = innvestigate.analyzer.Gradient(model_wo_sm)
# Applying the analyzer
analysis = gradient_analyzer.analyze(val.loc[[row_num]])

print(analysis)
plt.plot(analysis[0])
plt.show()
#plt.savefig("baseline2.png")

