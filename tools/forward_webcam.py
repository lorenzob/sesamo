# USAGE
# python read_frames_fast.py --video videos/jurassic_park_intro.mp4

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import sys
import os
import cv2
import base64
import threading
import traceback
import RemoteForwardServer
import numpy as np

from profilehooks import profile

from tornado import gen
from tornado.ioloop import IOLoop
from tornado.websocket import websocket_connect

from imutils.video import FPS

waitLock = threading.Lock()
wait = 0

recognitionHost=sys.argv[1]

def onMessage(msg):
    
    global wait
    
    try:
    
        if msg == None:
            print("WARN - onMessage: " + str(msg) + " (wait: " + str(wait))
            return
        
        with waitLock:
            print("onMessage: " + str(msg) + " (wait: " + str(wait))
            if "PROCESSED" in msg:
                wait += -1
    except Exception, e:
        print str(e)
        traceback.print_exc()
    
@gen.engine
def sendMessage(frame, identity):

    global wait
    
    try:
        print("shape " + str(np.shape(frame)))

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 55]
        cnt = cv2.imencode('.jpg', frame, encode_param)[1]
        dataURL = "data:image/jpeg;base64,"+ base64.b64encode(cnt)
    
        if wsc is not None:
            #img = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD//gATQ3JlYXRlZCB3aXRoIEdJTVD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wgARCAAKAAoDAREAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAABwUG/8QAFwEAAwEAAAAAAAAAAAAAAAAAAQMEBv/aAAwDAQACEAMQAAABS037kOmrqN4dT//EABoQAAICAwAAAAAAAAAAAAAAAAIFAAQDEzX/2gAIAQEAAQUCY2LdFiOFcAoeJqCf/8QAHxEAAQQABwAAAAAAAAAAAAAAAQACAwQFERIyNFGB/9oACAEDAQE/AZLMxmEp3IvtuOZBWI8weLW7tf/EABoRAAICAwAAAAAAAAAAAAAAAAACERIBAzP/2gAIAQIBAT8Bwi1qQhp5kH//xAAhEAAABAUFAAAAAAAAAAAAAAAAAQIEAwURFDIhMTOCkf/aAAgBAQAGPwJnLmbG4lkbljV2BJJaKFpkFdhiXg//xAAaEAADAQADAAAAAAAAAAAAAAAAAREhMUHx/9oACAEBAAE/Id03tXPd6gpxSkC248Mf/9oADAMBAAIAAwAAABBzD//EABsRAAICAwEAAAAAAAAAAAAAAAABESExYZHB/9oACAEDAQE/EKOUiK8HyYd4HhyGx0//xAAZEQEBAAMBAAAAAAAAAAAAAAABABExUYH/2gAIAQIBAT8QIxqBMCW32w5f/8QAHBABAAMAAgMAAAAAAAAAAAAAAQARITFBccHw/9oACAEBAAE/ECubkm676o3eYZappgFHcAIEWw+WfPep/9k="
            #control_ws.write_message('{"type":"FRAME","dataURL":"' + dataURL + '","identity":"' + identity + '"}', binary=False)
            msgData = '{"type":"FRAME","dataURL":"' + dataURL + '","identity":"' + identity + '"}'
            with waitLock:
                #wsc.enqueueMessage(msgData)
                wsc.enqueueMessage(cnt.tobytes())
                wait += 1
                print("sendMessage - wait: " + str(wait))
        else:
            print("no connection")
    except Exception, e:
        print str(e)
        traceback.print_exc()

from multiprocessing.dummy import Pool
pool = Pool(processes=1)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#@profile
def processFrame(frame, gray):
    
    
    try:
        #print("processFrame started (wait: " + str(wait))
    
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.4,
            minNeighbors=4,  #Higher value results in less detections but with higher quality.
            minSize=(80, 80),
            flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            print("Found {0} faces!".format(len(faces)))
            sendMessage(frame, "test")
            #IOLoop.instance().add_callback(lambda: sendMessage(frame, "test"))
            #IOLoop.instance().start()
    except Exception, e:
        print str(e)
        traceback.print_exc()
        

def processFrameCallback(frame):
    #print("processFrame completed")
    pass

def startProcessFrame(frame, gray):

    try:    
        open_callback_function = \
            lambda f: processFrameCallback(f)
        
        pool.apply_async(
            processFrame,
            args=[frame, gray],
            callback=open_callback_function)
    except Exception, e:
        print str(e)
        traceback.print_exc()


wsc = None

def forwardFrames():
    
    global wsc

    try:

        wsc = RemoteForwardServer.RemoteForwardServer('ws://{}:9003'.format(recognitionHost))
        wsc.startAsync(onMessage)
        time.sleep(1)
    
        fvs = VideoStream(src=0).start()
        time.sleep(1.0)
     
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
    
        # loop over frames from the video file stream
        fps = FPS().start()
        localFps = FPS().start()
        frameCount = 0
        print("Streaming started")
        lastFrame = None 
        while True:
            
            frame = fvs.read()

            if frame is None:
                print("End of video.")
                # TODO: reconnect
                break
    
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if lastFrame is not None:
                sameFrame = np.array_equal(frame, lastFrame)
                if sameFrame:
                    continue
                
            lastFrame = frame
    
            localFps.update()
    
            frameCount += 1
            if frameCount % 100 == 0:
                localFps.stop()
                localCount = localFps.fps()
                print("[INFO] Local approx. FPS: {:.2f}".format(localCount))
                localFps = FPS().start()
                
                fps.stop()
                count = fps.fps()
                if count > 1:
                    print("[INFO] approx. FPS: {:.2f}".format(count))
                fps = FPS().start()
        
            if wait > 2:
                continue
            
            #if len(faces) > 0:
            # Usare pool?
            #IOLoop.instance().add_callback(lambda: sendMessage(frame, "test"))
            #IOLoop.instance().start()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #startProcessFrame(frame, gray)
            processFrame(frame, gray)
    
            fps.update()
            
        	# show the frame and update the FPS counter
        	#cv2.imshow("Frame", frame)
        	#cv2.waitKey(1)
        
        # stop the timer and display FPS information
        
        # do a bit of cleanup
        #cv2.destroyAllWindows()
        fvs.stop()
        
    except Exception, e:
        print str(e)
        traceback.print_exc()


def log_except_hook(*exc_info):
    
    import logging
    import traceback
    
    try:
        text = "".join(traceback.format_exception(*exc_info))
        print("Unhandled exception " + text)
    except Exception, e:
        print str(e)

if __name__ == '__main__':

    import sys
    sys.excepthook = log_except_hook
    
    forwardFrames()

