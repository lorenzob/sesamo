import numpy as np
import cv2
import os
import sys
import openface

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

script_path = os.path.dirname(os.path.realpath(__file__))
predictor_model = script_path + "/shape_predictor_68_face_landmarks.dat"

# Take the image file name from the command line
file_name = "/media/trz/DATA/sesamo-data/video/dance/new/" + sys.argv[1]

# Create a HOG face detector using the built-in dlib class

print("Inizio")
cap = cv2.VideoCapture(file_name)
#cap = cv2.VideoCapture(0)
print(cap)
align = openface.AlignDlib("../openface/models/dlib/shape_predictor_68_face_landmarks.dat")

frameNum=0
fileName="./video/video-caps-{}/frame-".format(sys.argv[2])
skipFrames=int(sys.argv[3])
capturedFrames=0
ensure_dir(fileName)
while(cap.isOpened()):
    ret, frame = cap.read()

    if frame is None:
        print("End of video.")
        break

    if frameNum % skipFrames == 0:

        rgbImg = frame
        
        bb = align.getAllFaceBoundingBoxes(rgbImg)
        if len(bb) == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
            continue
            
        #if len(bb) > 1:
        #    sys.stdout.write('!')
        #    sys.stdout.flush()
        #    continue
    
        for box in bb:    # anche se ne ho una sola
            cv2.imwrite(fileName + str(frameNum) + ".jpg", rgbImg)
            sys.stdout.write('+')
            sys.stdout.flush()
            capturedFrames += 1

    frameNum += 1

    #cv2.imshow('frame', frame)
    #print(".")

    #if capturedFrames == 100:
    #    break
        
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()

# cv2.destroyAllWindows()
