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
from AVRecorder import AudioRecorder
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
#from PIL import Image
#import numpy as np
import os
import StringIO
import urllib
import base64
import time
import copy

#from sklearn.decomposition import PCA
#from sklearn.grid_search import GridSearchCV
#from sklearn.manifold import TSNE
#from sklearn.svm import SVC
#from sklearn.preprocessing import LabelEncoder

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

#import pickle

#import openface

import requests

raspberry_url = 'http://192.168.2.117:3000'

cwd = os.path.dirname(os.path.realpath(__file__))
userDataDir = cwd + "/web-identities"

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=9002,
                    help='WebSocket Port')
args = parser.parse_args()


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

class OpenFaceServerProtocol(WebSocketServerProtocol):

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
        
        print("onMessage")
        
        raw = payload.decode('utf8')
        msg = json.loads(raw)
        if msg['type'] == "NULL":
            # handshake iniziale ELIMINARE
            self.sendMessage('{"type": "NULL"}')
        print("Received {} message of length {}.".format(
            msg['type'], len(raw)))
        if msg['type'] == "START_RECORDING":

            asynch_matches = msg['matches']
            
            print("asynch_matches: " + str(asynch_matches))
            
            # Se fatto asincrono va in Segmentation Fault
            #self.startAsyncSpeakerRecognition(asynch_matches)
            
            file_audio = self.asyncSpeakerRecognition(asynch_matches)
            #print("file_audio " + file_audio + "<")
            confidence = self.speakerRecognitionCallback(file_audio)

            msg = {
                "type": "AUDIO_MATCH_RESULT",
                "confidence": confidence,
            }
            self.sendMessage(json.dumps(msg))
            
            print("Fine FRAME")
            

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
        
    audio_recorder = None

    def asyncSpeakerRecognition(self, matches):
    
        import AVRecorder
 
        if self.audio_recorder == None:
            self.audio_recorder = AudioRecorder()
       
        print("asyncSpeakerRecognition started...")
        print("Invoke start recording")
        try:
            file_name = self.audio_recorder.start(10)
            self.audio_recorder.waitForRecordingCompletion()
            return (file_name, matches)
        except Exception as ex:
            print "Exception in asyncTraining " + str(ex) 
            print "Unexpected error:", sys.exc_info()[0]            
                
        return "error"
                
    def speakerRecognitionCallback(self, audio_file_and_video_matches):
        
        #signal = self.backend.filter(Main.FS, signal)
        #    if len(signal) > 50:
        
        import SpeakerRecognition
        
        print "Callback: " + str(audio_file_and_video_matches)
        audio_file = str(audio_file_and_video_matches[0])

        if audio_file == None:
            return
        
        print "-- Callback: " + str(audio_file)
        matches = audio_file_and_video_matches[1]
        print "-- Callback: " + str(matches)
        
        # TODO: Singolo subject
        match = matches[0]
        print "-- Callback: " + str(match)
        name = match[0] 
        print "-- Callback: " + str(name)
        
        print("check voice for " + name)

        print("cwd " + cwd + "<")
        print("audio_file " + audio_file + "<")

        audio_model = name.lower() + ".out"
        confidence = SpeakerRecognition.task_predict(cwd + "/" + audio_file, audio_model)
        print "-- SpeakerRecognition res: " + str(confidence)

        return confidence


if __name__ == '__main__':
    
    log.startLogging(sys.stdout)

    factory = WebSocketServerFactory("ws://localhost:{}".format(args.port),
                                     debug=False)
    factory.protocol = OpenFaceServerProtocol

    reactor.listenTCP(args.port, factory)
    reactor.run()
