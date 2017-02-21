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

rb_url = 'http://192.168.0.7:3000'

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
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9000,
                    help='WebSocket Port')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=True)

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
        self.training = True
        self.people = []
        self.svm = None
        if args.unknown:
            self.unknownImgs = np.load("./examples/web/unknown.npy")
        
        # Qui non va bene, scatta sulla prima connessione
        # self.doTraining()

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = False

    def onOpen(self):
        print("WebSocket connection open.")


    def asyncTraining(self):
        print("asyncTraining started")
        self.doTraining()
        print("asyncTraining finished")


    def trainingCallback(self, msg):
        print "Callback: " + str(msg)

    def doTraining(self):
        
        print "Start training from: " + self.userDataDir
        print "Loading data from disk..."
        # load the data from file system
        X, y, self.knownUsers = self.trainFromFolder(net, self.userDataDir)
        print "Loading done."
        self.le = LabelEncoder().fit(self.knownUsers)

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
            self.svm = SVC(C=1, kernel='linear', probability=True)
        else:
            param_grid = [{'C':[1, 10, 100, 1000], 
                    'kernel':['linear']}, 
                     {'C':[1, 10, 100, 1000], 
                    'gamma':[0.001, 0.0001], 
                    'kernel':['rbf']}]
            self.svm = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
    
        print "Fitting data..."
        self.svm.fit(X, y)
        print "Fitting done"
        from scipy.stats import itemfreq
        freq = itemfreq(y)
        userFreq = []
        for x in freq:
            userFreq.append(str(x))

        msg = {
            "type":"IDENTITIES", 
            "identities":userFreq}
        self.sendMessage(json.dumps(msg))
        
        #print(userFreq)
        print "Training done"
        return userFreq

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        print("Received {} message of length {}.".format(
            msg['type'], len(raw)))
        if msg['type'] == "NULL":
            # handshake iniziale
            self.sendMessage('{"type": "NULL"}')
        elif msg['type'] == "FRAME":
            # singolo frame della webcam
            self.processFrame(msg['dataURL'], msg['identity'])
            self.sendMessage('{"type": "PROCESSED"}')
        elif msg['type'] == "TRAINING":
            self.training = msg['val']
            
            if self.training:
                self.currentTrainingSubject = msg['extra']
                self.trainingFramesCount = 0
            else:
                self.currentTrainingSubject = None
                
                userFreq = ["Training started, please wait..."]
                
                # userFreq = self.doTraining()
                # asynch training                
                new_callback_function = \
                    lambda new_name: self.trainingCallback(new_name)
                
                self.pool.apply_async(
                    self.doTraining,
                    args=[],
                    callback=new_callback_function
                )
                print("asyncTraining invoked")

                msg = {
                    "type":"IDENTITIES", 
                    "identities":userFreq}
                self.sendMessage(json.dumps(msg))
                
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))
    
    def trainFromFolder(self, net, userDataDir):

        msg = {
            "type": "IDENTITIES",
            "identities": "Updating network data..."
        }
        # Non appare...
        #self.sendMessage(json.dumps(msg))
        
        X = []
        y = []
        utenti = []
    
        for userName in os.listdir(userDataDir):
            print "\n Reading images for" + userName

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


    #win = dlib.image_window()
    lastSaveTime = 0
    trainingFramesCount = 0

    def processFrame(self, dataURL, identity):
        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        imgdata = base64.b64decode(dataURL[len(head):])
        imgF = StringIO.StringIO()
        imgF.write(imgdata)
        imgF.seek(0)
        img = Image.open(imgF)

        # A quanto pare dalla webcam arriva alla roverscia
        # e lo rigira per poterlo visualizzare giusto
        buf = np.fliplr(np.asarray(img))
        
        annotatedFrame = np.copy(buf)

        # Trovare volti (da capire se serve il BGR2RGB)
        rgbImg = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
        #self.win.set_image(annotatedFrame) #raw_input("Press Enter to continue...")
        
        # align
        bbs = align.getAllFaceBoundingBoxes(rgbImg)

        if self.training:
            
            nome = self.currentTrainingSubject
            text = "Training: {}".format(nome);
            cv2.putText(annotatedFrame, text, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                        color=(0, 255, 0), thickness=1)

            if len(bbs) > 1:
                msg = {
                    "type": "IDENTITIES",
                    "identities": ["Training problem: too many people, only {} should be present".format(nome)]
                }
                self.sendMessage(json.dumps(msg))
                #return
            elif len(bbs) == 0:
                msg = {
                    "type": "IDENTITIES",
                    "identities": ["Training problem: nobody's there. {} should be present".format(nome)]
                }
                self.sendMessage(json.dumps(msg))
                #return
            else:
                for box in bbs:   # in realta' ne ho uno solo
                    
                    bl = (box.left(), box.bottom()); tr = (box.right(), box.top())
                    cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 204),
                                  thickness=2)
                    
                    alignedFace = align.align(
                            SAMPLES_IMG_SIZE,
                            rgbImg,
                            box,
                            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    
                    # Per debug
                    rgbFace = cv2.cvtColor(alignedFace, cv2.COLOR_BGR2RGB)
                    #self.win.set_image(rgbFace) #raw_input("Press Enter to continue...")
            
                    # Immaginina
                    l_img = annotatedFrame
                    s_img = rgbFace
                    x_offset = l_img.shape[1] - 100
                    y_offset = l_img.shape[0] - 100
                    l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
                
                    # salvo immaginetta
                    if time.time() - self.lastSaveTime > 0.25:
                        fileName = "{}/{}/{}.jpg".format(self.userDataDir, nome, time.time())
                        print("Saving img to: {}".format(fileName))
                        # outBgr = cv2.cvtColor(alignedFace, cv2.COLOR_RGB2BGR)
                        ensure_dir(fileName)
                        cv2.imwrite(fileName, alignedFace)
                        self.lastSaveTime = time.time()
                        self.trainingFramesCount = self.trainingFramesCount + 1
                    else:
                        print("Skipping frame save: too close")
                
                frames = self.trainingFramesCount
                msg = {
                    "type": "IDENTITIES",
                    "identities": ["Training {}...".format(nome), "Captured frames: {} (we need about 100)".format(frames)]
                }
                self.sendMessage(json.dumps(msg))
        else:

            # Riconoscimento
            lineShift = 0
            for box in bbs:
                alignedFace = align.align(
                        SAMPLES_IMG_SIZE,
                        rgbImg,
                        box,
                        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

                bl = (box.left(), box.bottom()); tr = (box.right(), box.top())
                cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 204),
                              thickness=2)
                
                # Devo convetirla RGB?
                alignedFace = cv2.cvtColor(alignedFace, cv2.COLOR_BGR2RGB)
                #self.win.set_image(alignedFace) #raw_input("Press Enter to continue...")
            
                # Mini picture
                l_img = annotatedFrame
                s_img = alignedFace
                x_offset = l_img.shape[1] - 100
                y_offset = l_img.shape[0] - 100
                l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img
            
                if self.svm:
                    
                    rep = net.forward(alignedFace).reshape(1, -1)
                    # Il reshape serve perche' chiedo una predizione sola
                    # qui sarebbe probabilmente piu' giusto accumulare
                    # i volti e farne una sola(?)
                    predictions = self.svm.predict_proba(rep).ravel()
                    
                    maxI = np.argmax(predictions)
                    confidence = predictions[maxI]

                    text = "{} (confidence {})".format(self.le.inverse_transform(maxI), confidence)
                    cv2.putText(annotatedFrame, text, (5, 20 + lineShift),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                                color=(0, 255, 0), thickness=1)
                    lineShift = lineShift + 15;

            msg = {
                "type": "IDENTITIES",
                "identities": ["AAA", "BBB"]
            }
            # self.sendMessage(json.dumps(msg))

        plt.figure()
        plt.imshow(annotatedFrame)
        plt.xticks([])
        plt.yticks([])

        imgdata = StringIO.StringIO()
        plt.savefig(imgdata, format='png')
        imgdata.seek(0)
        content = 'data:image/png;base64,' + \
            urllib.quote(base64.b64encode(imgdata.buf))
        msg = {
            "type": "ANNOTATED",
            "content": content
        }
        plt.close()
        
        #from random import randint
        #if randint(0,9) > 5:
        self.sendMessage(json.dumps(msg))

    def open(self, openFlag):

        print("Remote open command: '{0}'".format(openFlag))
        
        payload = {'open': openFlag}

        # GET with params in URL
        r = requests.get(rb_url, params=payload)
        
        r.text
        r.status_code

if __name__ == '__main__':
    log.startLogging(sys.stdout)

    factory = WebSocketServerFactory("ws://localhost:{}".format(args.port),
                                     debug=False)
    factory.protocol = OpenFaceServerProtocol

    reactor.listenTCP(args.port, factory)
    reactor.run()
