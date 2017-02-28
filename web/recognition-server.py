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
# Applicazione REST con per il riconoscimento a partire da una foto
# 
# 1. Riceve una foto e ritorna nome, confidenza e location
#
# 2. Riceve un pkl con la rete corrente e si aggiorna rispetto a quella
#
# 3. Comando per forzare il reload della rete
#
# 4. Info sulla rete corrente (data caricamento, utenti, ecc.)
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
import copy

from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pickle

import openface

import requests

raspberry_url = 'http://192.168.2.117:3000'

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
parser.add_argument('--port', type=int, default=9003,
                    help='WebSocket Port')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                              cuda=False)
svm = None
knownUsers = []
le = LabelEncoder().fit(knownUsers)
svmDefinitionFile = "current-classifier.pkl"

cwd = os.path.dirname(os.path.realpath(__file__))
userDataDir = cwd + "/web-identities"


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def loadDefaultSVMData():
    svmFile = userDataDir + "/" + svmDefinitionFile
    loadSVMData(svmFile)

def loadSVMData(fileName):
    print("Reading data from: " + fileName)
    with open(fileName, 'r') as f:
        (lEnc, clf) = pickle.load(f)
    global svm 
    global le
    svm = clf
    le = lEnc
    print("Reading data completed: " + str(svm))

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
    
    def __init__(self):
        self.images = {}
        self.training = True
        self.people = []
        ensure_dir(userDataDir)

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = False

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        if msg['type'] == "NULL":
            # handshake iniziale ELIMINARE
            self.sendMessage('{"type": "NULL"}')
        print("Received {} message of length {}.".format(
            msg['type'], len(raw)))
        if msg['type'] == "FRAME":
            # singolo frame della webcam
            self.processFrame(msg['dataURL'], msg['identity'])
            self.sendMessage('{"type": "PROCESSED"}')
        elif msg['type'] == "UPDATE_SVM":
            print("UPDATE_SVM")
            updateLocalSVMDefinitionOnDisk()
        elif msg['type'] == "RELOAD_SVM":
            print("RELOAD_SVM")
            loadDefaultSVMData()
        elif msg['type'] == "SVM_INFOS":
            print("SVM_INFOS")
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))


    def startAsyncSpeakerRecognition(self, matches):
        new_callback_function = \
            lambda new_name: self.speakerRecognitionCallback(new_name)
        
        self.pool.apply_async(
            self.asyncSpeakerRecognition,
            args=[matches],
            callback=new_callback_function)
        
    def asyncSpeakerRecognition(self, matches):

        from websocket import create_connection
        ws = create_connection("ws://localhost:9004/")
        print "Sending 'Hello, World'..."
        
        msg = {
            "type": "START_RECORDING",
            "matches": matches
        }
        
        ws.send(json.dumps(msg))
        print "Sent. Receiving..."
        result =  ws.recv()
        print "Received '%s'" % result
        ws.close()
        
        return "error"
                
    def speakerRecognitionCallback(self, audio_file_and_video_matches):
        
        print("check voice for " + audio_file_and_video_matches)


    def updateLocalSVMDefinitionOnDisk(self):

        # TODO: tutto da implementare, l'idea cmq e' quella qui sotto

        head = "data:image/jpeg;base64,"
        assert(dataURL.startswith(head))
        pklData = base64.b64decode(dataURL[len(head):])
        
        # salvo subito su disco in modo che rimanga 
        # disponibile dopo un riavvio.
        # TODO: caricarlo all'avvio
        svmNewFile = userDataDir + "/" + svmDefinitionFile + ".new"
        with open(svmNewFile, "w") as f:
            f.write(pklData)

        self.loadSVMData(svmNewFile)

        #print("Reading data from: " + svmNewFile)
        #with open(svmNewFile, 'r') as f:
        #    (le, clf) = pickle.load(f)

        # Faccio un backup del file precedente
        #now = time.time()
        #os.rename(svmDefinitionFile, svmDefinitionFile + ".bak-" + str(now))
        os.rename(svmNewFile, userDataDir + "/" + svmDefinitionFile)
        
    def processFrame(self, dataURL, identity):
        
        if svm is None: # gestire meglio con un errore a monte
            print("No svm")
            return
        
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

        # Riconoscimento
        matches = []
        usersInFrame = []
        for box in bbs:
            alignedFace = align.align(
                    SAMPLES_IMG_SIZE,
                    rgbImg,
                    box,
                    landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                
            rep = net.forward(alignedFace).reshape(1, -1)
            # Il reshape serve perche' chiedo una predizione sola
            # qui sarebbe probabilmente piu' giusto accumulare
            # i volti e farne una sola(?) Si puo'?
            predictions = svm.predict_proba(rep).ravel()
            
            maxI = np.argmax(predictions)
            confidence = predictions[maxI]

            nome = le.inverse_transform(maxI)
            # text = "{} (confidence {})".format(nome, confidence)

            location = [box.left(), box.bottom(), box.right(), box.top()]
            matches.append([nome, confidence, location])
            
            text = "{} (confidence {})".format(nome, confidence)
            usersInFrame.append(text)
            
            #self.sendMessage(json.dumps(msg))
        print(matches)
        
        if matches:
            # Ho un match, inizio a registrare
            asynch_matches = copy.deepcopy(matches)
            self.startAsyncSpeakerRecognition(asynch_matches)

        msg = {
            "type": "IDENTITIES",
            "identities": usersInFrame
        }
        self.sendMessage(json.dumps(msg))
#        plt.figure()
#        plt.imshow(annotatedFrame)
#        plt.xticks([])
#        plt.yticks([])

#        imgdata = StringIO.StringIO()
        #plt.savefig(imgdata, format='png')
#        plt.savefig(imgdata, format="jpg", dpi=75, quality=15)
#        imgdata.seek(0)
#        content = 'data:image/png;base64,' + \
#            urllib.quote(base64.b64encode(imgdata.buf))
        #plt.close()

        # Ogni match e' fatto da [nome, confidenza, location]
        # dove location e' [xL, yBottom, xR, yTop] 
        # print(matches)
        msg = {
            "type": "MATCHES",
            "identities": matches
        }
        self.sendMessage(json.dumps(msg))
        

    def open(self, openFlag):

        print("Remote open command: '{0}'".format(openFlag))
        
        payload = {'open': openFlag}

        # GET with params in URL
        r = requests.get(raspberry_url, params=payload)
        
        r.text
        r.status_code

if __name__ == '__main__':
    
    log.startLogging(sys.stdout)

    loadDefaultSVMData()

    factory = WebSocketServerFactory("ws://localhost:{}".format(args.port),
                                     debug=False)
    factory.protocol = OpenFaceServerProtocol

    reactor.listenTCP(args.port, factory)
    reactor.run()
