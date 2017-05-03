#!/usr/bin/env python2

import sys
import dlib
from skimage import io

# Take the image file name from the command line
file_name = sys.argv[1]
#file_name = "Farrell-Smith.jpg"
#file_name = "three.jpg"
#file_name = "crowd3.jpg"
#file_name = "small group2.jpg"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

win = dlib.image_window()

# Load the image into an array
image = io.imread(file_name)

# Run the HOG face detector on the image data.
# The result will be the bounding boxes of the faces in our image.
detected_faces = face_detector(image, 0)

print("I found {} faces in the file {}".format(len(detected_faces), file_name))

# Open a window on the desktop showing the image
win.set_image(image)

# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):

    # Detected faces are returned as an object with the coordinates 
    # of the top, left, right and bottom edges
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
    print("    Size {}x{}".format(face_rect.right()-face_rect.left(), face_rect.bottom()-face_rect.top()))

    # Draw a box around each face we found
    win.add_overlay(face_rect)
            
# Wait until the user hits <enter> to close the window            
dlib.hit_enter_to_continue()