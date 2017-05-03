#!/usr/bin/env python2
#


# To run:
# classifier.py infer classifier.pkl image.jpg

from __future__ import division

import time
from keras.metrics import precision

def split_path(dirname):
    
    size = None
    if "." in dirname:
        userUnderTest, size = dirname.split(".")
    else:
        userUnderTest = dirname
        
    return userUnderTest, size

start = time.time()

import argparse
import cv2
import os
import sys
import pickle

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

fileDir = os.path.dirname(os.path.realpath(__file__))

script_path = os.path.dirname(os.path.realpath(__file__))
dlibModelDir = script_path
openfaceModelDir = script_path

threshold = 0.8

le = None
clf = None

class Stats:
    
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    def recordExpectedMatch(self, true_pos, false_neg):
        self.true_pos += true_pos
        self.false_neg += false_neg
        
    def recordExpectedNoMatch(self, false_pos, true_neg):
        self.true_neg += true_neg
        self.false_pos += false_pos

    def count(self):
        return (self.true_pos + self.false_neg) + (self.true_neg + self.false_pos)
    
    def accuracy(self):
        acc = -1
        if self.count() > 0:
            acc = (self.true_pos + self.true_neg) / self.count()
        return acc
        
    def f1(self):
        f1 = -1
        if self.precision() + self.recall() > 0:
            f1 = 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())
        return f1
        
    def precision(self):
        precision = -1
        if self.true_pos + self.false_pos > 0:
            precision = self.true_pos / (self.true_pos + self.false_pos)
        return precision
            
    def recall(self):
        recall = -1
        if self.true_pos + self.false_neg > 0:
            recall = self.true_pos / (self.true_pos + self.false_neg)
        return recall

    def values(self):
        return (self.accuracy(), self.f1(), self.precision(), self.recall())


# estraggo le feature dall'immagine
def getRep(imgPath, multiple=True, faceFragment=False):
    start = time.time()
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    reps = []

    if faceFragment:
        rep = net.forward(bgrImg)
        reps.append((100, rep))
    else:
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    
        if args.verbose:
            print("  + Original size: {}".format(rgbImg.shape))
        if args.verbose:
            print("Loading the image took {} seconds.".format(time.time() - start))
    
        start = time.time()
    
        if multiple:
            bbs = align.getAllFaceBoundingBoxes(rgbImg)
        else:
            bb1 = align.getLargestFaceBoundingBox(rgbImg)
            bbs = [bb1]
        if len(bbs) == 0 or (not multiple and bb1 is None):
            raise Exception("Unable to find a face: {}".format(imgPath))
        if args.verbose:
            print("Face detection took {} seconds.".format(time.time() - start))
    
        for bb in bbs:
            start = time.time()
            # Qui non passa i landmark, nell'altro si'
            alignedFace = align.align(
                args.imgDim,
                rgbImg,
                bb,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                raise Exception("Unable to align image: {}".format(imgPath))
            if args.verbose:
                print("Alignment took {} seconds.".format(time.time() - start))
                print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))
    
            start = time.time()
    
            # estraggo le feature dall'immagine
            rep = net.forward(alignedFace)
            if args.verbose:
                print("Neural network forward pass took {} seconds.".format(
                    time.time() - start))
            reps.append((bb.center().x, rep))
            
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps


def infer(imgs, multiple=False, faceFragment=False):
    
    for img in imgs:
        # print("=== {} ===".format(img))
        reps = getRep(img, multiple, faceFragment)
        if len(reps) > 1:
            print("List of faces in image from left to right")
        for r in reps:
            rep = r[1].reshape(1, -1)
            bbx = r[0]
            start = time.time()
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            confidence = predictions[maxI]
            if args.verbose:
                print("Prediction took {} seconds.".format(time.time() - start))
            if multiple:
                print("Predict {} @ x={} with {:.3f} confidence.".format(person, bbx,
                                                                         confidence))
            #else:
                 #print("Predict {} with {:.3f} confidence.".format(person, confidence))
                 #sys.stdout.write('.')
            if isinstance(clf, GMM):
                dist = np.linalg.norm(rep - clf.means_[maxI])
                print("  + Distance from the mean: {}".format(dist))
                
            return (person, confidence)

def test_classifier(classifier, imgsFolder, expected, shouldMatch):
    
    if shouldMatch:
        print("## Positive validation: classifier: {}, data: {}, expected: {}".format(classifier, imgsFolder, expected))
    else:
        print("## Negative validation: classifier: {}, data: {}, expected: {}".format(classifier, imgsFolder, expected))
        
    imgs = [imgsFolder + "/" + f for f in os.listdir(imgsFolder) if os.path.isfile(imgsFolder + "/" + f)]
    #print(imgs)

    match = 0
    noMatch = 0
    for img in imgs:
        
        person, confidence = infer([img], args.multi, args.faceFragment)

        validMatch = None
        if expected == 'unknown':
            # a me va bene sia se viene classificato come 
            # sconosciuto sia se non matcha nessun altro
            validMatch = (person == expected or confidence <= threshold)
        else:
            validMatch = (person == expected and confidence > threshold)

        if validMatch:
            match += 1
            if not shouldMatch:
                print("*** False positive: NOT expected match({}), predicted '{}({})' testing folder: {} - file: {}".format(expected, person, confidence, imgsFolder, img))
        else:
            noMatch += 1
            if shouldMatch:
                if person == expected:
                    print("*** False negative (low threshold): expected match({}), predicted '{}({})' testing folder: {} - file: {}".format(expected, person, confidence, imgsFolder, img))
                else:
                    print("*** False negative: expected match({}), predicted '{}({})' testing folder: {} - file: {}".format(expected, person, confidence, imgsFolder, img))
            
    return match, noMatch
        
def get_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

if __name__ == '__main__':
    
    import fnmatch

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=128)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    subparsers = parser.add_subparsers(dest='mode', help="Mode")

    inferParser = subparsers.add_parser(
        'infer', help='Predict who an image contains from a trained classifier.')
    inferParser.add_argument(
        'classifiersFolder',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')
    inferParser.add_argument('validationData', type=str,
                             help="Input image.")
    inferParser.add_argument('--multi', help="Infer multiple faces in image",
                             action="store_true")
    inferParser.add_argument('--faceFragment', help="Frammenti volto gia' estratti",
                             action="store_false", default=True)
    inferParser.add_argument('--unknownData', help="Test ulteriore con unknown",)
    inferParser.add_argument('--fullUnknownTest', help="Testo anche unknown nel trainingData",
                             action="store_true", default=False)

    args = parser.parse_args()
    if args.verbose:
        print("Argument parsing and import libraries took {} seconds.".format(
            time.time() - start))

    start = time.time()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                                      cuda=True)

    if args.verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(
            time.time() - start))
        start = time.time()

    print("Validation folder " + args.validationData)

    classifiers = []
    for root, dirnames, filenames in os.walk(args.classifiersFolder):
        for filename in fnmatch.filter(filenames, '*.pkl'):
            classifiers.append(os.path.join(root, filename))
    dirnames = get_subdirectories(args.validationData)

    for classifier in classifiers:

        print("##############################")
        print("### Classifier under test {} ({})".format(classifier, args.validationData))
        print("###>>> START({}, {}, {}, {})".format(args.classifiersFolder, classifier, threshold, args.validationData))
        print("##############################")
        startClass = time.time()
        
        with open(classifier, 'r') as f:
            (le, clf) = pickle.load(f)
        
        stats = Stats()

        if not args.unknownData is None:
            
            # argh: overrido il valore passato e poi lo riassegno
            cliSpecifiedValue = args.faceFragment
            args.faceFragment = True
            
            match, noMatch = test_classifier(classifier, args.unknownData, "unknown", shouldMatch=True)
            stats.recordExpectedMatch(match, noMatch)
            print("###>>> unknown tp:{}, tn:{}, fp:{}, fn:{}, acc:{}, f1:{}, prec:{}, rec:{}".format(stats.true_pos, stats.true_neg, stats.false_pos, stats.false_neg, stats.accuracy(), stats.f1(), stats.precision(), stats.recall()))

            args.faceFragment = cliSpecifiedValue
            
        print("Users: {}".format(dirnames))
        for dirname in dirnames:
            #print("User folder " + dirname)
    
            userUnderTest, size = split_path(dirname)
            print("### User under test {}({})".format(userUnderTest, size))

            if args.unknownData is not None and not args.fullUnknownTest:
                userUnderTest = "unknown"
                    
            # itero sugli stessi folder
            for compareWithDir in dirnames:
    
                comparisonUser, comparisonSize = split_path(compareWithDir)
    
                print("Comparing {} with {}".format(userUnderTest, comparisonUser))

                if (args.unknownData is None) \
                        and (userUnderTest == "unknown" or comparisonUser == "unknown") \
                        and not args.fullUnknownTest:
                    print("Skipping unknown test")
                    continue
    
                if userUnderTest == comparisonUser:
                    # qui deve matchare
                    if size == comparisonSize:
                        match, noMatch = test_classifier(classifier, args.validationData + "/" + compareWithDir, userUnderTest, shouldMatch=True)
                        stats.recordExpectedMatch(match, noMatch)
                    else:
                        print("skip: " + compareWithDir)
                else:
                    # qui deve fallire
                    match, noMatch = test_classifier(classifier, args.validationData + "/" + compareWithDir, userUnderTest, shouldMatch=False)
                    stats.recordExpectedNoMatch(match, noMatch)
    
                print("###>>> Curr user {}: tp:{}, tn:{}, fp:{}, fn:{}, acc:{}, f1:{}, prec:{}, rec:{}".format(userUnderTest, stats.true_pos, stats.true_neg, stats.false_pos, stats.false_neg, stats.accuracy(), stats.f1(), stats.precision(), stats.recall()))

            print("###>>> After user {}: tp:{}, tn:{}, fp:{}, fn:{}, acc:{}, f1:{}, prec:{}, rec:{}".format(userUnderTest, stats.true_pos, stats.true_neg, stats.false_pos, stats.false_neg, stats.accuracy(), stats.f1(), stats.precision(), stats.recall()))
            
            if not args.unknownData is None and not args.fullUnknownTest:
                break

                    
        print("###>>> FINAL({}, {}, {}, {}, {}): tp:{}, tn:{}, fp:{}, fn:{}, acc:{}, f1:{}, prec:{}, rec:{}".format(args.classifiersFolder, classifier, threshold, args.validationData, args.unknownData, stats.true_pos, stats.true_neg, stats.false_pos, stats.false_neg, stats.accuracy(), stats.f1(), stats.precision(), stats.recall()))
        print("###>>> Took: {}".format(time.time() - startClass))

    print("Done.")
