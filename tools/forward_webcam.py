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

from tornado import gen
from tornado.ioloop import IOLoop
from tornado.websocket import websocket_connect


@gen.engine
def connect():
    
    global control_ws
    
    url = "ws://localhost:9003"

    print("connect")    
    control_ws = yield websocket_connect(url, None)
    print("connect ok: " + str(control_ws))
    
    IOLoop.instance().stop()

@gen.engine
def sendMessage(frame, identity):

    cnt = cv2.imencode('.jpg', frame)[1]
    dataURL = "data:image/jpeg;base64,"+ base64.b64encode(cnt)

    if control_ws is not None:
        #img = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD//gATQ3JlYXRlZCB3aXRoIEdJTVD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wgARCAAKAAoDAREAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAABwUG/8QAFwEAAwEAAAAAAAAAAAAAAAAAAQMEBv/aAAwDAQACEAMQAAABS037kOmrqN4dT//EABoQAAICAwAAAAAAAAAAAAAAAAIFAAQDEzX/2gAIAQEAAQUCY2LdFiOFcAoeJqCf/8QAHxEAAQQABwAAAAAAAAAAAAAAAQACAwQFERIyNFGB/9oACAEDAQE/AZLMxmEp3IvtuOZBWI8weLW7tf/EABoRAAICAwAAAAAAAAAAAAAAAAACERIBAzP/2gAIAQIBAT8Bwi1qQhp5kH//xAAhEAAABAUFAAAAAAAAAAAAAAAAAQIEAwURFDIhMTOCkf/aAAgBAQAGPwJnLmbG4lkbljV2BJJaKFpkFdhiXg//xAAaEAADAQADAAAAAAAAAAAAAAAAAREhMUHx/9oACAEBAAE/Id03tXPd6gpxSkC248Mf/9oADAMBAAIAAwAAABBzD//EABsRAAICAwEAAAAAAAAAAAAAAAABESExYZHB/9oACAEDAQE/EKOUiK8HyYd4HhyGx0//xAAZEQEBAAMBAAAAAAAAAAAAAAABABExUYH/2gAIAQIBAT8QIxqBMCW32w5f/8QAHBABAAMAAgMAAAAAAAAAAAAAAQARITFBccHw/9oACAEBAAE/ECubkm676o3eYZappgFHcAIEWw+WfPep/9k="
        control_ws.write_message('{"type":"FRAME","dataURL":"' + dataURL + '","identity":"' + identity + '"}', binary=False)
    else:
        print("no connection")
    
    IOLoop.instance().stop()


def forwardFrames():

    fvs = VideoStream(src=0).start()
    time.sleep(1.0)
    
    # loop over frames from the video file stream
    print("Streaming started")
    while True:
    
    	frame = fvs.read()
    
    	if frame is None:
    		print("End of video.")
    		break
    
    	rgbImg = frame
    
        IOLoop.instance().add_callback(lambda: sendMessage(frame, "test"))
        IOLoop.instance().start()
        
    	# show the frame and update the FPS counter
    	cv2.imshow("Frame", frame)
    	cv2.waitKey(1)
    
    # stop the timer and display FPS information
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    fvs.stop()


if __name__ == '__main__':
    
    IOLoop.instance().run_sync(connect)
    IOLoop.instance().start()
    
    forwardFrames()
