import tornado
from tornado.websocket import websocket_connect
import time
import threading
import numpy as np
import io
import sys
import os

import Queue

class RemoteForwardServer(object):

    # Remote open pool
    from multiprocessing.dummy import Pool
    pool = Pool(processes=1)

    framesQueue = Queue.Queue()
    onMessageCallback = None

    def __init__(self, url):
        self.url = url
        self.ioloop = tornado.ioloop.IOLoop.current()

    def start(self):
        
        print("start connect")
        f = websocket_connect(
            self.url,
            self.ioloop,
            callback=self.on_connected,
            on_message_callback=self.onMessageCallback)
        
        print("start loop")
        self.ioloop.start()
        print("fine start")

    conn = None
    buffer = None
    bufferSize = 0
    bufferLock = threading.Lock()

    def on_message(self, msg):
        print("on_message")
        print msg
    
    def on_connected(self, f):
        print("on_connected")
        try:
            self.conn = f.result()
            
            identity = "test"
            #conn.write_message('{"type":"FRAME","dataURL":"' + img + '","identity":"' + identity + '"}', binary=False)
            
        except Exception, e:
            print str(e)
            self.ioloop.stop()
        print("fine")

    def remoteOpenCallback(self, openFlag):
        print("startup completed " + str(openFlag))
    
    def startAsync(self, onMessageCallback):
        print("startup started")
        
        self.onMessageCallback = onMessageCallback
        
        open_callback_function = \
            lambda new_name: self.remoteOpenCallback(new_name)
        
        self.pool.apply_async(
            self.remoteOpen,
            args=[1],
            callback=open_callback_function)

    def enqueueMessage(self, msgData):
        
        import struct
            
        try:
            print(type(msgData))
            print(len(msgData))

            # Header: {2:magic} {2:version} {8:timestamp} {4:source}            
            with self.bufferLock:
                
                if self.buffer is None:
                    self.buffer = io.BytesIO()
                    self.buffer.write("SM01TIMESTMPSRC1")
                    
                print("self.buffer size: {} ".format(self.buffer.tell()))
                    
                self.buffer.write(struct.pack('>H', len(msgData)))
                self.buffer.write(msgData)
                
                self.bufferSize += 1
                
                #self.buffer.seek(0)
                #var = self.buffer.read()
                #self.buffer = None
                #self.ioloop.add_callback(self.sendMsg, var)
        except Exception, e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def clearAllBuffer(self):
        with self.bufferLock:
            self.buffer = None
            self.bufferSize = 0

    def readBufferSize(self):
        with self.bufferLock:
            return self.bufferSize

    def flushAllMessages(self):
        try:
            with self.bufferLock:
                
                if self.buffer is not None:
                    print("flushAllMessages count: {}, size: {} ".format(self.bufferSize, self.buffer.tell()))
                    self.buffer.seek(0)
                    var = self.buffer.read()
                    self.buffer = None
                    self.ioloop.add_callback(self.sendMsg, var)
                    
                    self.bufferSize = 0

        except Exception, e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def sendMsg(self, msgData):
        #print("sendMsg " + str(len(msgData)))
        
        if self.conn is not None:
            #img = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD//gATQ3JlYXRlZCB3aXRoIEdJTVD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wgARCAAKAAoDAREAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAABwUG/8QAFwEAAwEAAAAAAAAAAAAAAAAAAQMEBv/aAAwDAQACEAMQAAABS037kOmrqN4dT//EABoQAAICAwAAAAAAAAAAAAAAAAIFAAQDEzX/2gAIAQEAAQUCY2LdFiOFcAoeJqCf/8QAHxEAAQQABwAAAAAAAAAAAAAAAQACAwQFERIyNFGB/9oACAEDAQE/AZLMxmEp3IvtuOZBWI8weLW7tf/EABoRAAICAwAAAAAAAAAAAAAAAAACERIBAzP/2gAIAQIBAT8Bwi1qQhp5kH//xAAhEAAABAUFAAAAAAAAAAAAAAAAAQIEAwURFDIhMTOCkf/aAAgBAQAGPwJnLmbG4lkbljV2BJJaKFpkFdhiXg//xAAaEAADAQADAAAAAAAAAAAAAAAAAREhMUHx/9oACAEBAAE/Id03tXPd6gpxSkC248Mf/9oADAMBAAIAAwAAABBzD//EABsRAAICAwEAAAAAAAAAAAAAAAABESExYZHB/9oACAEDAQE/EKOUiK8HyYd4HhyGx0//xAAZEQEBAAMBAAAAAAAAAAAAAAABABExUYH/2gAIAQIBAT8QIxqBMCW32w5f/8QAHBABAAMAAgMAAAAAAAAAAAAAAQARITFBccHw/9oACAEBAAE/ECubkm676o3eYZappgFHcAIEWw+WfPep/9k="
            #identity = "test"
            #self.conn.write_message('{"type":"FRAME","dataURL":"' + img + '","identity":"' + identity + '"}', binary=False)
            
            #self.conn.write_message(msgData)
            #print("len " + str(len(msgData)))
            #print("shape " + str(np.shape(msgData)))
            self.conn.write_message(msgData, binary=True)
    
    def remoteOpen(self, openFlag):
        print("remoteOpen started " + str(openFlag))

        self.start()  # Qui si blocca e parte ioloop

        while True:
            #print("add callback")
            try:
                #self.enqueueMessage(1)
                #print("callback added")
                #time.sleep(0)
                pass
            except Exception, e:
                print str(e)
        #self.ioloop.start()


def log_except_hook(*exc_info):
    
    import logging
    import traceback
    
    try:
        text = "".join(traceback.format_exception(*exc_info))
        print("Unhandled exception " + text)
    except Exception, e:
        print str(e)



def bytes_to_int(bytes):
    result = 0

    for b in bytes:
        result = result * 256 + int(b)

    return result

def int_to_bytes(value, length):
    result = []

    for i in range(0, length):
        result.append(value >> (i * 8) & 0xff)

    result.reverse()

    return result



if __name__=='__main__':

    import sys
    sys.excepthook = log_except_hook
    
    wsc = RemoteForwardServer('ws://ec2-52-41-184-155.us-west-2.compute.amazonaws.com:9003')
    wsc.startAsync(wsc.on_message)
    #wsc.start()
    time.sleep(1)

    #wsc.enqueueMessage(1)
    #wsc.enqueueMessage(1)
    #wsc.enqueueMessage(1)
    #wsc.enqueueMessage(1)
    
    time.sleep(4)

