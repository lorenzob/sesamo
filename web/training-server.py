#!/usr/bin/env python2
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#####################################################################
#
# Applicazione REST per il training della rete a partire dalle foto
# 
# 1. avvio indicizzazione (callback di notifica fine indicizzazione?)
#
# 2. status indicizzazione
#
# 3. riceve pacchetto foto(?)
#
#####################################################################


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

import argparse
import cv2
import imagehash
import json
from PIL import Image
import numpy as np
import os
import StringIO
import urllib
import base64
import time

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import openface

import requests

#rb_url = 'http://192.168.2.117:3000'

SAMPLES_IMG_SIZE = 96


modelDir = os.path.join(fileDir, '.', '.', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=SAMPLES_IMG_SIZE)
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9002,
                    help='WebSocket Port')

args = parser.parse_args()

#align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=False)

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    

class Face:

    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5]
        )


class OpenFaceServerProtocol(WebSocketServerProtocol):

    # Temp: solo per l'immagine di controllo
    import dlib
    import time

    from multiprocessing.dummy import Pool
    pool = Pool(processes=1)
    
    cwd = os.path.dirname(os.path.realpath(__file__))
    userDataDir = cwd + "/web-identities"
    print("userDataDir: {}".format(userDataDir))

    status = "ready"
    currentTrainingSubject = None
    knownUsers = []
    le = LabelEncoder().fit(knownUsers)

    def __init__(self):
        self.images = {}
        self.training = False
        self.people = []
        self.svm = None
        ensure_dir(self.userDataDir)
        
        # Qui non va bene, scatta sulla prima connessione
        # self.doTraining()

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = False

    def onOpen(self):
        print("WebSocket connection open.")
        msg = {
            "type":"IDENTITIES", 
            "identities": ["Ready"]}
        self.sendMessage(json.dumps(msg))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))
    
    def asyncTraining(self):
        print("asyncTraining started...")
        self.doTraining()
        print("asyncTraining finished")

    def trainingCallback(self, userFreq):
        print "Callback: " + str(userFreq)

    def doTraining(self):
        
        newLe = None
        newSvm = None
        
        print "Start training from: " + self.userDataDir
        print "Loading data from disk..."
        # load the data from file system
        X, y, self.knownUsers = self.trainFromFolder(net, self.userDataDir)
        print "Loading done."
        newLe = LabelEncoder().fit(self.knownUsers)

        msg = {
            "type":"IDENTITIES", 
            "identities":["Fitting data..."]}
        self.sendMessage(json.dumps(msg))
        
        # train the network
        # see: http://scikit-learn.org/stable/modules/svm.html
        
        # In un commento dicevano che dai loro test 
        # il semplice SVC funziona bene quanto il 
        # piu' complesso GridSearchCV
        if 1 == 1:
            newSvm = SVC(C=1, kernel='linear', probability=True)
        else:
            param_grid = [{'C':[1, 10, 100, 1000], 
                    'kernel':['linear']}, 
                     {'C':[1, 10, 100, 1000], 
                    'gamma':[0.001, 0.0001], 
                    'kernel':['rbf']}]
            newSvm = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
    
        print "Fitting data..."
        newSvm.fit(X, y)
        print "Fitting done."
        
        #Non indispensabile
        print "Printing stats..."
        from scipy.stats import itemfreq
        freq = itemfreq(y)
        userFreq = []
        for x in freq:
            userFreq.append(str(x))
        print "Stats: " + str(userFreq)

        try:
            import pickle

            fName = "{}/current-classifier.pkl".format(self.userDataDir)
            print("Saving classifier to '{}'".format(fName))
            with open(fName, 'w') as f:
                pickle.dump((newLe, newSvm), f)
        except Exception as ex:
            print "Exception in asyncTraining " + str(ex) 
            print "Unexpected error:", sys.exc_info()[0]            
        

        msg = {
            "type":"IDENTITIES", 
            "identities":userFreq}
        self.sendMessage(json.dumps(msg))

        self.le = newLe
        self.svm = newSvm
        
        self.training = False
        
        print "Training done."
        return userFreq

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        print("Received {} message of length {}.".format(
            msg['type'], len(raw)))
        if msg['type'] == "NULL":
            # handshake iniziale
            self.sendMessage('{"type": "NULL"}')
        if msg['type'] == "START_TRAINING":
            
            if self.training:
                print("Training already running")
                msg = {
                    "type":"IDENTITIES", 
                    "identities": ["Training already running"]}
                self.sendMessage(json.dumps(msg))
            else:
                
                self.training = True
                
                # asynch training
                new_callback_function = \
                    lambda new_name: self.trainingCallback(new_name)
                
                self.pool.apply_async(
                    self.doTraining,
                    args=[],
                    callback=new_callback_function
                )
                print("asyncTraining invoked")

                text = ["Training started, please wait..."]
                msg = {
                    "type":"IDENTITIES", 
                    "identities":text}
                self.sendMessage(json.dumps(msg))
        elif msg['type'] == "TRAINING_STATUS":
            print("ecc.")
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def trainFromFolder(self, net, userDataDir):

        try:

            msg = {
                "type": "IDENTITIES",
                "identities": ["Updating network data..."]
            }
            self.sendMessage(json.dumps(msg))
            
            X = []
            y = []
            utenti = []
        
            userDataFolders = os.walk(userDataDir).next()[1]
            for userName in userDataFolders:
                print "\n Reading images for " + userName
    
                utenti.append(userName)
                imgDir = userDataDir + "/" + userName
                frameCount = 0
                files = os.listdir(imgDir)
                for userImg in files:
                    
                    sys.stdout.write('.')
                    msg = {
                        "type":"IDENTITIES", 
                        "identities":["Training {} [{}/{}] (please wait)...".format(userName, frameCount, len(files))]}
                    self.sendMessage(json.dumps(msg))
                    
                    # Take the image file name from the command line
                    file_name = imgDir + "/" + userImg
                    # Load the image
                    image = cv2.imread(file_name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
                    rep = net.forward(image)
                    X.append(rep)
                    y.append(userName)
                    
                    frameCount = frameCount+1
            
            # Fine training
            X = np.vstack(X)
            y = np.array(y)
            
            print "\n Fine training"
            return X, y, utenti
        
        except Exception as ex:
            print "Exception in asyncTraining " + str(ex) 
            print "Unexpected error:", sys.exc_info()[0]          


if __name__ == '__main__':
    log.startLogging(sys.stdout)

    factory = WebSocketServerFactory("ws://localhost:{}".format(args.port),
                                     debug=False)
    factory.protocol = OpenFaceServerProtocol

    reactor.listenTCP(args.port, factory)
    reactor.run()
