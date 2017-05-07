import cv2
import sys
import time

# Get user supplied values
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
for i in range(10):
    start = time.time()
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.4,
        minNeighbors=4,  #Higher value results in less detections but with higher quality.
        minSize=(80, 80),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )
    
    end = time.time()
    
    print("Found {0} faces! (time: {1})".format(len(faces), (end-start)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)