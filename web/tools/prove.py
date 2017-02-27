import sys
import dlib
import cv2
import openface

import os
import datetime

import pickle


from sklearn.preprocessing import LabelEncoder
#---



# temp
#import argparse
import numpy as np

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

inizio = datetime.datetime.now()

cwd = os.path.dirname(os.path.realpath(__file__))
print("Current working dir: {}".format(cwd))

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '.', '.', 'models')
openfaceModelDir = os.path.join(modelDir, 'openface')
nn4Model = default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')

dlibModelDir = os.path.join(modelDir, 'dlib')
shape_predictor = default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")

net = openface.TorchNeuralNet(nn4Model, imgDim=96,
                              cuda=False)

print("Training:")


list = [1, 2, 3, 4]

#pkl = pickle.dumps(list)

with open("test.pkl", 'w') as f:
    pickle.dump(list, f)

# print(pkl)
fileStr = open("test.pkl").read()

# res = pickle.loads(pkl)
res = pickle.loads(fileStr)

print(res)

import time
now = time.time()
os.rename("test.pkl", "test2.pkl" + str(now))

#with open(args.classifierModel, 'r') as f:
#    (le, clf) = pickle.load(f)  # le - label and clf - classifer




#repName = "{}/identities/rep_{}".format(dir_path, time.time())
#np.save(repName, rep)

# Save to file

# Temp
#import pandas as pd
#from operator import itemgetter
#
#fname = "{}/../../generated-embeddings/labels.csv".format(cwd)
#labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
#labels = map(itemgetter(1),
#             map(os.path.split,
#                 map(os.path.dirname, labels)))  # Get the directory.
#le = LabelEncoder().fit(labels)
# Fine Temp



