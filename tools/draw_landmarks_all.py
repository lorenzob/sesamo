#!/usr/bin/env python2

import sys
import os
import fnmatch
import dlib
from skimage import io

# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
script_path = os.path.dirname(os.path.realpath(__file__))
predictor_model = script_path + "/shape_predictor_68_face_landmarks.dat"

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

win = dlib.image_window()

file_name = sys.argv[1]

matches = []
for root, dirnames, filenames in os.walk(file_name):
    for filename in fnmatch.filter(filenames, '*'):
        matches.append(os.path.join(root, filename))

for fIdx, file_name in enumerate(matches):

    # Load the image
    image = io.imread(file_name)
    
    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)
    
    print("Found {} faces in the image file {}".format(len(detected_faces), file_name))
    
    # Show the desktop window with the image
    win.clear_overlay()
    win.set_image(image)
    
    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
    
        # Detected faces are returned as an object with the coordinates 
        # of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
        print("    Size {}x{}".format(face_rect.right()-face_rect.left(), face_rect.bottom()-face_rect.top()))
    
        # Draw a box around each face we found
        win.add_overlay(face_rect)
    
        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image, face_rect)
    
        # Draw the face landmarks on the screen.
        win.add_overlay(pose_landmarks)
                
    dlib.hit_enter_to_continue()
    

print("Done")