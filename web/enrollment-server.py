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
# Applicazione web con cui l'utente salva le proprie foto
#
# Le foto vengono salvate in locale e a fine training trasferite
# altrove. A quel punto viene dato il comando di reindicizzazione
#
# In questo momento fa queste cose:
# 1. riceve il nome dell'utente che sta effettuando il training
# 2. riceve un frame, verifica che ci sia un singolo volto, lo 
#    estrae/normalizza e lo salva
# 3. gestisce la fine del training trasferendo i file
# 4. spedisce verso il client un'immagine in cui si visibile il riquadro 
#    verde di riconoscimento
#
# Da riorganizzare come: 
# unire 1 e 2, 
# 3 come prima (gestendo piu' training attivi, spedendo solo le immagini giuste)
# 4 spedire le coordinate del rettangolo, non un'immagine
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
#parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
#                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
#parser.add_argument('--imgDim', type=int,
#                    help="Default image dimension.", default=SAMPLES_IMG_SIZE)
#parser.add_argument('--cuda', action='store_false')
#parser.add_argument('--unknown', type=bool, default=False,
#                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9001,
                    help='WebSocket Port')

args = parser.parse_args()

align = openface.AlignDlib(args.dlibFacePredictor)
#net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
#                              cuda=args.cuda)

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

    currentTrainingSubject = None

    def __init__(self):
        self.images = {}
        self.training = True
        self.people = []
        self.svm = None

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = False

    def onOpen(self):
        print("WebSocket connection open.")

        msg = {
            "type":"IDENTITIES", 
            "identities": ["Ready"]}
        self.sendMessage(json.dumps(msg))

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
                nome = self.currentTrainingSubject
                self.currentTrainingSubject = None
                
                msg = {
                    "type":"IDENTITIES", 
                    "identities": ["Enrollment completed for {}. New frames: {}".format(nome, self.trainingFramesCount)]}
                self.sendMessage(json.dumps(msg))
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))
    
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

        # A quanto pare dalla webcam arriva alla rovescia
        # e lo rigira per poterlo visualizzare giusto
        buf = np.fliplr(np.asarray(img))
        
        annotatedFrame = np.copy(buf)

        # Trovare volti (da capire se serve il BGR2RGB)
        rgbImg = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
        #self.win.set_image(annotatedFrame) #raw_input("Press Enter to continue...")
        
        # align
        bbs = align.getAllFaceBoundingBoxes(rgbImg)

        matches = []
        if self.training:
            
            nome = self.currentTrainingSubject
            text = "Enrolling: {}".format(nome);
            cv2.putText(annotatedFrame, text, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                        color=(0, 255, 0), thickness=1)

            if len(bbs) > 1:
                msg = {
                    "type": "IDENTITIES",
                    "identities": ["Enrollment problem: too many people, only {} should be present".format(nome)]
                }
                self.sendMessage(json.dumps(msg))
            elif len(bbs) == 0:
                msg = {
                    "type": "IDENTITIES",
                    "identities": ["Enrollment problem: nobody's there. {} should be present".format(nome)]
                }
                self.sendMessage(json.dumps(msg))
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

                    location = [box.left(), box.bottom(), box.right(), box.top()]
                    matches.append([nome + "(?)", 0, location])
                
                frames = self.trainingFramesCount
                msg = {
                    "type": "IDENTITIES",
                    "identities": ["Enrolling {}...".format(nome), "Captured frames: {} (we need about 100)".format(frames)]
                }
                self.sendMessage(json.dumps(msg))
                                
        else:
            # Non durante il training

            lineShift = 0
            for box in bbs:

                bl = (box.left(), box.bottom()); tr = (box.right(), box.top())
                cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 204),
                              thickness=2)

                location = [box.left(), box.bottom(), box.right(), box.top()]
                matches.append(["New user", 0, location])

        msg = {
            "type": "MATCHES",
            "identities": matches
        }
        self.sendMessage(json.dumps(msg))


        #plt.figure()
        #plt.imshow(annotatedFrame)
        #plt.xticks([])
        #plt.yticks([])

        #TODO: vedere di spedirla super compressa?
        # http://stackoverflow.com/questions/10784652/png-options-to-produce-smaller-file-size-when-using-savefig
        #imgdata = StringIO.StringIO()
        #plt.savefig(imgdata, format="jpg", dpi=75, quality=15)

        #imgdata.seek(0)
        #content = 'data:image/png;base64,' + \
        #    urllib.quote(base64.b64encode(imgdata.buf))
        #msg = {
        #    "type": "ANNOTATED",
        #    "content": content
        #}
        #plt.close()
        
        #self.sendMessage(json.dumps(msg))

if __name__ == '__main__':
    log.startLogging(sys.stdout)

    factory = WebSocketServerFactory("ws://localhost:{}".format(args.port),
                                     debug=False)
    factory.protocol = OpenFaceServerProtocol

    reactor.listenTCP(args.port, factory)
    reactor.run()
