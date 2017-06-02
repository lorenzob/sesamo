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
import traceback
import threading

from profilehooks import profile

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

from imutils.video import FPS

import pickle

import openface

import requests

import RecognitionService
    


#raspberry_url = 'http://192.168.0.6:3000'

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
parser.add_argument('--cuda', action='store_false', default=False)
parser.add_argument('--headless', action='store_true', default=False)
parser.add_argument('--unknown', type=bool, default=False,
                    help='Try to predict unknown people')
parser.add_argument('--port', type=int, default=9003,
                    help='WebSocket Port')
parser.add_argument('--onRecognitionWebhook', default=None,
                    help='"Open the door" webhook')


args = parser.parse_args()

knownUsers = []
le = LabelEncoder().fit(knownUsers)
svmDefinitionFile = "current-classifier.pkl"

cwd = os.path.dirname(os.path.realpath(__file__))
userDataDir = cwd + "/web-identities"

raspberry_url=args.onRecognitionWebhook


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

class OpenFaceServerProtocol(WebSocketServerProtocol):

    # Temp: solo per l'immagine di controllo
    import dlib
    import time
    
    #win = dlib.image_window()
    
    recognitionService = None
    
    recognitionService = RecognitionService.RecognitionService(args)
    recognitionService.loadDefaultSVMData()
    
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

    framesQueueLock = threading.RLock()
    framesQueue = 0

    def processFrameCompleted(self, temp):
        try:
            with self.framesQueueLock:
                self.framesQueue += -1
                print("processFrame completed " + str(temp) + " - Pending: " + str(self.framesQueue))
            self.sendMessage('{"type": "PROCESSED"}')
        except Exception, e:
            print("### EXC: processFrameCompleted: ", str(e))
            traceback.print_exc()

    def onMessage(self, payload, isBinary):

        import struct

        print("onMessage")
        try:
            print("onMessage (" + str(isBinary) + " - " + str(len(payload)))
        
            if isBinary:
    
                #print(type(payload))
                rawData = bytearray(payload)
                #print(type(rawData))
                npdata = np.asarray(rawData)
                #print(type(npdata))
                
                headerLen = 16
                header = npdata[0:headerLen]
                #print(str(header))

                sizeLen = 2                
                currPos = headerLen
                print("Pre: " + str(currPos) + "/" + str(npdata.size))
                while currPos < npdata.size:
                    
                    #print(str(currPos) + "/" + str(npdata.size))
                    
                    sizeEnd = currPos+sizeLen
                    size = struct.unpack('>H', npdata[currPos:sizeEnd])[0]
                    imgData = npdata[sizeEnd:sizeEnd+size]
                    #print(len(imgData))
                    #print(type(imgData))
                
                    with self.framesQueueLock:
                        self.framesQueue += 1
                        print("Pending: " + str(self.framesQueue))
                        
                        if self.framesQueue < 4:
                            self.recognitionService.startAsyncProcessFrame(np.array(imgData).tostring(), "test", self.processFrameCompleted, binary=True)
                        else:
                            self.processFrameCompleted("dropped")
                    
                    currPos = sizeEnd+size
                    print("Post: " + str(currPos) + "/" + str(npdata.size))
                #if size > 0:
                #    with open("prova-buffer.jpeg", "w") as f:
                #        f.write(imgData)
                
                #img = cv2.imdecode(imgData, cv2.IMREAD_UNCHANGED)

                #self.win.set_image(img)

                #print("Invoke start process")
                
                #self.sendMessage('{"type": "PROCESSED"}')

                return
    
            print("Non binary")
            
            raw = payload.decode('utf8')
            msg = json.loads(raw)
            if msg['type'] == "NULL":
                # handshake iniziale ELIMINARE
                print("Send NULL")
                self.sendMessage('{"type": "NULL"}')
            print("Received {} message of length {}.".format(
                msg['type'], len(raw)))
            if msg['type'] == "FRAME":
                # singolo frame della webcam
                #self.startAsyncProcessFrame(msg['dataURL'], msg['identity'], binary=False)
                print("Send PROCESSED (non bin)")
                self.sendMessage('{"type": "PROCESSED"}')
            elif msg['type'] == "UPDATE_SVM":
                print("UPDATE_SVM")
                updateLocalSVMDefinitionOnDisk()
            elif msg['type'] == "RELOAD_SVM":
                print("RELOAD_SVM")
                recognitionService.loadDefaultSVMData()
                
                msg = {
                    "type": "IDENTITIES",
                    "identities": ["User data reloaded"]
                }
                print("Send IDS")
                self.sendMessage(json.dumps(msg))
                
            elif msg['type'] == "SVM_INFOS":
                print("SVM_INFOS")
            else:
                print("Warning: Unknown message type: {}".format(msg['type']))
        except Exception, e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("### EXC: onMessage: ", exc_type, fname, exc_tb.tb_lineno)
            traceback.print_exc()
    
    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

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

def init_yappi():
  import atexit
  import yappi

  print('[YAPPI START]')
  yappi.set_clock_type('wall')
  yappi.start()

  @atexit.register
  def finish_yappi():
    print('[YAPPI STOP]')

    yappi.stop()

    print('[YAPPI WRITE]')

    stats = yappi.get_func_stats()

    for stat_type in ['pstat', 'callgrind', 'ystat']:
      print('writing /tmp/pants.{}'.format(stat_type))
      stats.save('/tmp/pants.{}'.format(stat_type), type=stat_type)

    print('\n[YAPPI FUNC_STATS]')

    print('writing /tmp/pants.func_stats')
    with open('/tmp/pants.func_stats', 'wb') as fh:
      stats.print_all(out=fh)

    print('\n[YAPPI THREAD_STATS]')

    print('writing /tmp/pants.thread_stats')
    tstats = yappi.get_thread_stats()
    with open('/tmp/pants.thread_stats', 'wb') as fh:
      tstats.print_all(out=fh)

    print('[YAPPI OUT]')
        

if __name__ == '__main__':
    
    #init_yappi()
    
    log.startLogging(sys.stdout)

    factory = WebSocketServerFactory("ws://localhost:{}".format(args.port),
                                     debug=False)
    factory.protocol = OpenFaceServerProtocol

    reactor.listenTCP(args.port, factory)
    reactor.run()
