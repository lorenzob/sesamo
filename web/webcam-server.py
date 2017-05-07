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

import requests

from tornado import gen
from tornado.ioloop import IOLoop
from tornado.websocket import websocket_connect

#rb_url = 'http://192.168.2.117:3000'

SAMPLES_IMG_SIZE = 96


parser = argparse.ArgumentParser()
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
        print("Received {} message of length {}.".format(msg['type'], len(raw)))
        if msg['type'] == "NULL":
            # handshake iniziale
            self.sendMessage('{"type": "NULL"}')
        elif msg['type'] == "FRAME":
            # singolo frame della webcam
            self.processFrame(msg['dataURL'], msg['identity'])
            self.sendMessage('{"type": "PROCESSED"}')
        else:
            print("Warning: Unknown message type: {}".format(msg['type']))

    def onClose(self, wasClean, code, reason):
        print("WebSocket connection closed: {0}".format(reason))
    
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

        cv2.imwrite("websocket.jpeg", annotatedFrame)
        
        IOLoop.instance().add_callback(lambda: forwardMessage(dataURL, identity))
        IOLoop.instance().start()

        msg = {
            "type": "IDENTITIES",
            "identities": ["Webcam image received: " + str(time.time())]
        }
        self.sendMessage(json.dumps(msg))

control_ws = None

@gen.engine
def connect():
    
    global control_ws
    
    url = "ws://192.168.0.5:9003"

    print("connect")    
    control_ws = yield websocket_connect(url, None)
    print("connect ok: " + str(control_ws))

    IOLoop.instance().stop()


@gen.engine
def forwardMessage(dataURL, identity):

    if control_ws is not None:
        #img = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD//gATQ3JlYXRlZCB3aXRoIEdJTVD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wgARCAAKAAoDAREAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAABwUG/8QAFwEAAwEAAAAAAAAAAAAAAAAAAQMEBv/aAAwDAQACEAMQAAABS037kOmrqN4dT//EABoQAAICAwAAAAAAAAAAAAAAAAIFAAQDEzX/2gAIAQEAAQUCY2LdFiOFcAoeJqCf/8QAHxEAAQQABwAAAAAAAAAAAAAAAQACAwQFERIyNFGB/9oACAEDAQE/AZLMxmEp3IvtuOZBWI8weLW7tf/EABoRAAICAwAAAAAAAAAAAAAAAAACERIBAzP/2gAIAQIBAT8Bwi1qQhp5kH//xAAhEAAABAUFAAAAAAAAAAAAAAAAAQIEAwURFDIhMTOCkf/aAAgBAQAGPwJnLmbG4lkbljV2BJJaKFpkFdhiXg//xAAaEAADAQADAAAAAAAAAAAAAAAAAREhMUHx/9oACAEBAAE/Id03tXPd6gpxSkC248Mf/9oADAMBAAIAAwAAABBzD//EABsRAAICAwEAAAAAAAAAAAAAAAABESExYZHB/9oACAEDAQE/EKOUiK8HyYd4HhyGx0//xAAZEQEBAAMBAAAAAAAAAAAAAAABABExUYH/2gAIAQIBAT8QIxqBMCW32w5f/8QAHBABAAMAAgMAAAAAAAAAAAAAAQARITFBccHw/9oACAEBAAE/ECubkm676o3eYZappgFHcAIEWw+WfPep/9k="
        img = dataURL
        control_ws.write_message('{"type":"FRAME","dataURL":"' + img + '","identity":"' + identity + '"}', binary=False)
    else:
        print("no connection")
    
    IOLoop.instance().stop()

if __name__ == '__main__':
    log.startLogging(sys.stdout)

    IOLoop.instance().run_sync(connect)
    IOLoop.instance().start()
    print("fine connect")

    factory = WebSocketServerFactory("ws://localhost:{}".format(args.port),
                                     debug=False)
    factory.protocol = OpenFaceServerProtocol

    reactor.listenTCP(args.port, factory)
    reactor.run()
    
    
