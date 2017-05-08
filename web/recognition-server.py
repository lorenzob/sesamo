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

raspberry_url = 'http://192.168.0.6:3000'

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
    
    import RecognitionService
    
    win = dlib.image_window()

    recognitionService = RecognitionService.RecognitionService()
    recognitionService.loadDefaultSVMData()

    from multiprocessing.dummy import Pool
    pool = Pool(processes=4)
    
    def __init__(self):
        self.images = {}
        self.training = True
        self.people = []
        ensure_dir(userDataDir)

    def onConnect(self, request):
        print("Client connecting: {0}".format(request.peer))
        self.training = False

    fps = None
    
    def onOpen(self):
        print("WebSocket connection open.")
        try:
            self.fps = FPS().start()
        except Exception, e:
            print str(e)

    frameCount = 0
    
    def processFrameCompleted(self, temp):
        print("AAA processFrame completed " + str(temp))
        try:
            self.sendMessage('{"type": "PROCESSED"}')
        except Exception, e:
            print str(e)

    def onMessage(self, payload, isBinary):

        print("onMessage (" + str(isBinary))
        
        try:
            self.frameCount += 1
            self.fps.update()
            if self.frameCount % 30 == 0:
                self.fps.stop()
                count = self.fps.fps()
                if count > 1:
                    print("[INFO] approx. FPS: {:.2f}".format(count))
                self.fps = FPS().start()

            if isBinary:
    
                try:
                    print("Binary")
                    rawData = bytearray(payload)
                    npdata = np.asarray(rawData)
                    img = cv2.imdecode(npdata, cv2.IMREAD_UNCHANGED)
    
                    #self.win.set_image(img)
    
                    print("Invoke start process")
                    
                    self.recognitionService.startAsyncProcessFrame(rawData, "test", self.processFrameCompleted, binary=True)
                    #self.sendMessage('{"type": "PROCESSED"}')
                except Exception, e:
                    print str(e)
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
                loadDefaultSVMData()
                
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
            print str(e)

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))

    def remoteOpenCallback(self, openFlag):
        print("remoteOpen completed " + str(openFlag))

    def startAsyncRemoteOpen(self, openFlag):
        open_callback_function = \
            lambda new_name: self.remoteOpenCallback(new_name)
        
        self.pool.apply_async(
            self.remoteOpen,
            args=[openFlag],
            callback=open_callback_function)

    def remoteOpen(self, openFlag):

        print("Remote open command: '{0}'".format(openFlag))
        
        payload = {'open': openFlag}

        # GET with params in URL
        r = requests.get(raspberry_url, params=payload)
        
        r.text
        r.status_code
        
    def startAsyncSpeakerRecognition(self, matches):
        new_callback_function = \
            lambda new_name: self.speakerRecognitionCallback(new_name)
        
        self.pool.apply_async(
            self.speakerRecognition,
            args=[matches],
            callback=new_callback_function)
        
    def speakerRecognition(self, matches):

        from websocket import create_connection
        ws = create_connection("ws://localhost:9004/")
        print "Sending 'Hello, World'..."
        
        msg = {
            "type": "START_RECORDING",
            "matches": matches
        }
        
        ws.send(json.dumps(msg))
        #print "Sent. Receiving..."
        result =  ws.recv()
        print "Audio result '%s'" % result
        ws.close()
        
        return "error"
                
    def speakerRecognitionCallback(self, audio_file_and_video_matches):
        print("check voice for " + audio_file_and_video_matches)

    def processFrameCallback(self, openFlag):
        print("processFrame completed " + str(openFlag))

    def startAsyncProcessFrame(self, dataURL, identity, binary):
        print("startAsyncProcessFrame")
        open_callback_function = \
            lambda new_name: self.remoteOpenCallback(new_name)
        
        self.pool.apply_async(
            self.processFrame,
            args=[dataURL, identity, binary],
            callback=self.processFrameCallback)

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
        
    #@profile
    def processFrame(self, dataURL, identity, binary):
        
        try:
        
            if svm is None: # gestire meglio con un errore a monte
                print("No svm")
                return
            
            if binary:
                print("Process binary image")
                imgF = StringIO.StringIO() #buffer where image is stored
                imgF.write(dataURL) #data is from the socket
                imgF.seek(0)
            else:
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
            
            #raw_input("Press Enter to continue...")
    
            #cv2.imwrite("forwarded.jpg", rgbImg)
            
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
                
                cv2.putText(annotatedFrame, text, (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                            color=(0, 255, 0), thickness=1)
                bl = (box.left(), box.bottom()); tr = (box.right(), box.top())
                cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 204),
                              thickness=1)
                
                #self.sendMessage(json.dumps(msg))
            print(matches)
            
            if matches:
                # Ho un match, inizio a registrare
                #self.startAsyncRemoteOpen(1)
                pass
            else:
                #self.startAsyncRemoteOpen(0)
                pass
    
            msg = {
                "type": "IDENTITIES",
                "identities": usersInFrame
            }
            self.sendMessage(json.dumps(msg))
            print("Send IDS (dentro process)")
    #        plt.figure()
    #        plt.imshow(annotatedFrame)
    #        plt.xticks([])
    #        plt.yticks([])
            self.win.set_image(annotatedFrame)
    
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
            print("Send MATCHES (dentro process)")
            
        except Exception, e:
            print "Process frame: " + str(e)
        
        self.sendMessage('{"type": "PROCESSED"}')
        print("Send PROCESSED (dentro process)")

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

    loadDefaultSVMData()

    factory = WebSocketServerFactory("ws://localhost:{}".format(args.port),
                                     debug=False)
    factory.protocol = OpenFaceServerProtocol

    reactor.listenTCP(args.port, factory)
    reactor.run()
