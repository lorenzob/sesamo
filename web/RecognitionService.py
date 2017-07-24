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
import threading
from datetime import datetime
import dlib
import time
import traceback
from MultiMatch import MultiMatch
from MultiMatch import Match

from profilehooks import profile
import Queue

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


class Face:

    def __init__(self, rep, identity):
        self.rep = rep
        self.identity = identity

    def __repr__(self):
        return "{{id: {}, rep[0:5]: {}}}".format(
            str(self.identity),
            self.rep[0:5]
        )


class RecognitionService:

    from multiprocessing.dummy import Pool

    svm = None
    knownUsers = []
    le = LabelEncoder().fit(knownUsers)
    svmDefinitionFile = "current-classifier.pkl"
    
    cwd = os.path.dirname(os.path.realpath(__file__))
    userDataDir = cwd + "/web-identities"

    fps = FPS().start()
    fpsLock = threading.Lock()
    lastFpsCount = 0
    frameCount = 0
    
    trackingCallback = None 
    onRecognitionCallback = None 
    remoteOpenPool = Pool(processes=1)
    
    POOL_SIZE=1
    pool = Pool(processes=POOL_SIZE)
        
    networks = Queue.Queue()
    
    matches = MultiMatch(0.9, 3)
    matchesLock = threading.Lock()
    
    win = None

    def __init__(self, args):
        self.images = {}
        self.training = True
        self.people = []
        
        print("cuda: {}".format(args.cuda))
        print("headless: {}".format(args.headless))
        self.headless = args.headless
        if not self.headless:
            self.win = dlib.image_window()

        for i in range(self.POOL_SIZE):
            print("Init pool element {}/{}...".format(i+1, self.POOL_SIZE))
            align = openface.AlignDlib(args.dlibFacePredictor)
            net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                                          cuda=args.cuda)
            self.networks.put((align, net))        

        if args.onRecognitionWebhook is not None:
            self.onRecognitionCallback = self.startAsyncRemoteOpen

    def setTrackingCallback(self, callback):
        self.trackingCallback = callback

    def loadDefaultSVMData(self):
        svmFile = self.userDataDir + "/" + self.svmDefinitionFile
        self.loadSVMData(svmFile)
    
    def loadSVMData(self, fileName):
        print("Reading data from: " + fileName)
        with open(fileName, 'r') as f:
            (lEnc, clf) = pickle.load(f)

        self.svm = clf
        self.le = lEnc
        print("Reading data completed: " + str(self.svm))
 
    def remoteOpenCallback(self, openFlag):
        print("remoteOpen completed " + str(openFlag))

    def startAsyncRemoteOpen(self, matches):
        
        print("Start asynch remote open...")
        
        open_callback_function = \
            lambda new_name: self.remoteOpenCallback(new_name)

        match = matches[0]
        openFlag = 1 if match[1] > 0.95 else 0
        
        self.remoteOpenPool.apply_async(
            self.remoteOpen,
            args=[openFlag],
            callback=open_callback_function)

    def remoteOpen(self, openFlag):

        try:

            print("Remote open command: '{0}'".format(openFlag))
            
            payload = {'open': openFlag}
    
            # GET with params in URL
            r = requests.get(raspberry_url, params=payload)
            
            r.text
            r.status_code
        except Exception, e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("### EXC: remoteOpen: ", exc_type, fname, exc_tb.tb_lineno)
            traceback.print_exc()
               
    def processFrameCallback(self, openFlag):
        print("internal processFrame completed (non usato) " + str(openFlag))

    def startAsyncProcessFrame(self, dataURL, identity, frameId, completion_callback, binary):
        #print("startAsyncProcessFrame")
        open_callback_function = \
            lambda new_name: completion_callback(new_name, frameId)
        
        self.pool.apply_async(
            self.processFrame,
            args=[dataURL, identity, frameId, binary],
            callback=open_callback_function)

        
    #@profile
    def processFrame(self, dataURL, identity, frameId, binary):
        
        import dlib
        
        try:
            print("processFrame " + frameId)
            
            start = time.time()
    
            align, net = self.networks.get_nowait()
        
            if self.svm is None: # gestire meglio con un errore a monte
                print("No svm")
                return
            
            if binary:
                #print("Process binary image")
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
            print("shape img: " + str(np.shape(img)))
    
            # A quanto pare dalla webcam arriva alla roverscia
            # e lo rigira per poterlo visualizzare giusto
            buf = np.fliplr(np.asarray(img))
            
            annotatedFrame = np.copy(buf)
    
            # Trovare volti (da capire se serve il BGR2RGB)
            rgbImg = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
            
            height, width = buf.shape[:2]
            scale = 2
            small = cv2.resize(buf, (width/scale, height/scale), interpolation = cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            
            # align
            bbs = align.getAllFaceBoundingBoxes(gray)
            
            # Riconoscimento
            currMatches = []
            usersInFrame = []
            for box in bbs:
                
                largeBox = dlib.rectangle(left=box.left()*scale, top=box.top()*scale, right=box.right()*scale, bottom=box.bottom()*scale)
                
                alignedFace = align.align(
                        SAMPLES_IMG_SIZE,
                        rgbImg,
                        largeBox,
                        landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                    
                rep = net.forward(alignedFace).reshape(1, -1)
                
                # Il reshape serve perche' chiedo una predizione sola
                # qui sarebbe probabilmente piu' giusto accumulare
                # i volti e farne una sola(?) Si puo'?
                predictions = self.svm.predict_proba(rep).ravel()
                
                maxI = np.argmax(predictions)
                confidence = predictions[maxI]
    
                nome = self.le.inverse_transform(maxI)
    
                location = [box.left(), box.bottom(), box.right(), box.top()]
                match = Match(nome, confidence, time.time(), location)
                #matches.append([nome, confidence, location])
                with self.matchesLock:
                    self.matches.record(match)
                
                text = "{} (confidence {})".format(nome, confidence)
                usersInFrame.append(text)
                
                if self.win is not None:
                    cv2.putText(annotatedFrame, text, (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                                color=(0, 255, 0), thickness=1)
                    
                    bl = (box.left()*scale, box.bottom()*scale); tr = (box.right()*scale, box.top()*scale)
                    cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 204),
                                  thickness=1)
                
                scaledLocation = [bl[0], bl[1], tr[0], tr[1]]
                currMatches.append([nome, confidence, scaledLocation])
                
                #self.sendMessage(json.dumps(msg))
            print("matches " + str(self.matches))

            with self.matchesLock:
                if self.matches.check() and self.onRecognitionCallback is not None:
                    self.onRecognitionCallback(matches)
                self.trackingCallback(currMatches, frameId)
            
            if self.win is not None:
                status = "{} - FPS: {:.2f}".format(datetime.now(), self.lastFpsCount)
                cv2.putText(annotatedFrame, status, (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                            color=(0, 255, 0), thickness=1)
                self.win.set_image(annotatedFrame)

            self.updateFPS()
            
            #print("time: " + str(time.time() - start))
        except Exception, e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("### EXC: processFrame: ", exc_type, fname, exc_tb.tb_lineno)
            traceback.print_exc()
        finally:
            self.networks.put((align, net))
        
    def updateFPS(self):
        self.fps.update()
        self.frameCount += 1
        if self.frameCount % 30 == 0:
            with self.fpsLock:
                self.fps.stop()
                count = self.fps.fps()
                if count > 0:
                    print("[INFO] approx. FPS: {:.2f}".format(count))
                self.lastFpsCount = count
                self.fps = FPS().start()

        
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
    
    #log.startLogging(sys.stdout)

    start = time.time()


    file = open("test-img.jpeg", 'r')
    data = bytearray(file.read())
    file.close()

    test = RecognitionService()
    test.loadDefaultSVMData()
    
    for i in range(10):
        test.startAsyncProcessFrame(data, "identity", True)

    test.pool.close()
    test.pool.join()

    print(time.time() - start)
    


    #factory = WebSocketServerFactory("ws://localhost:{}".format(args.port),
    #factory.protocol = OpenFaceServerProtocol

    #reactor.listenTCP(args.port, factory)
    #reactor.run()
