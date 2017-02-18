import sys
import dlib
import cv2
import openface

import os

import pickle


#---
'''
import os
import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

import txaio
txaio.use_twisted()

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from twisted.python import log
from twisted.internet import reactor

import cv2
import imagehash
import json
from PIL import Image
import os
import StringIO
import urllib
import base64
import time



import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import openface

import requests
'''
#---

# temp
#import argparse
import numpy as np


#from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
#from sklearn.manifold import TSNE
from sklearn.svm import SVC
#from sklearn.preprocessing import LabelEncoder

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#predictor_model = "shape_predictor_68_face_landmarks.dat"

# Take the image file name from the command line
#file_name = sys.argv[1]

# Create a HOG face detector using the built-in dlib class
#face_detector = dlib.get_frontal_face_detector()
#face_pose_predictor = dlib.shape_predictor(predictor_model)
#face_aligner = openface.AlignDlib(predictor_model)

cwd = os.path.dirname(os.path.realpath(__file__))
print("Current working dir: {}".format(cwd))

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', '..', 'models')
openfaceModelDir = os.path.join(modelDir, 'openface')
nn4Model = default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')

dlibModelDir = os.path.join(modelDir, 'dlib')
shape_predictor = default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")

# align = openface.AlignDlib(shape_predictor)
net = openface.TorchNeuralNet(nn4Model, imgDim=96,
                              cuda=False)

# Carica tutte le immagini
X = []
y = []

utenti = []

userDataDir = cwd + "/identities"
for userName in os.listdir(userDataDir):
    print(userName)
    utenti.append(userName)

    imgDir = userDataDir + "/" + userName
    for userImg in os.listdir(imgDir):

        #print(userImg)

        # Take the image file name from the command line
        file_name = imgDir + "/" + userImg
        
        # Load the image
        image = cv2.imread(file_name)
        
        #win = dlib.image_window()
        #win.set_image(image)
        #dlib.hit_enter_to_continue()
        
        #print("Forward " + userImg)
        rep = net.forward(image)
        #print("Forward done")
        
        X.append(rep)
        y.append(userName)


# Fine training

X = np.vstack(X)
y = np.array(y)

param_grid = [
    {'C': [1, 10, 100, 1000],
     'kernel': ['linear']},
    {'C': [1, 10, 100, 1000],
     'gamma': [0.001, 0.0001],
     'kernel': ['rbf']}
]
svm = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5).fit(X, y)

fName = "{}/my-classifier.pkl".format(cwd)
print("Saving classifier to '{}'".format(fName))
with open(fName, 'w') as f:
    pickle.dump((le, svm), f)

# Test

lorenzo = cv2.imread(cwd + "/identities/lorenzo/" + "1487172459.24.jpg")
kelly = cv2.imread(cwd + "/identities/kelly/" + "1487176992.41.jpg")
chen = cv2.imread(cwd + "/identities/chen/" + "1487176926.1.jpg")
pedro = cv2.imread(cwd + "/identities/pedro/" + "1487176896.14.jpg")

rep = net.forward(kelly)

predictions = svm.predict_proba(rep).ravel()

maxI = np.argmax(predictions)
confidence = predictions[maxI]

print("Utente {} (confidence: {})".format(utenti[maxI-1], confidence))

#repName = "{}/identities/rep_{}".format(dir_path, time.time())
#np.save(repName, rep)



