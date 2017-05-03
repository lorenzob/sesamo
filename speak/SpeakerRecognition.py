#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: speaker-recognition.py
# Date: Sun Feb 22 22:36:46 2015 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import argparse
import sys
import glob
import os
import itertools
import scipy.io.wavfile as wavfile

sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'gui'))
from gui.interface import ModelInterface
from gui.utils import read_wav
from filters.silence import remove_silence

def get_args():
    desc = "Speaker Recognition Command Line Tool"
    epilog = """
Wav files in each input directory will be labeled as the basename of the directory.
Note that wildcard inputs should be *quoted*, and they will be sent to glob.glob module.

Examples:
    Train (enroll a list of person named person*, and mary, with wav files under corresponding directories):
    ./speaker-recognition.py -t enroll -i "/tmp/person* ./mary" -m model.out

    Predict (predict the speaker of all wav files):
    ./speaker-recognition.py -t predict -i "./*.wav" -m model.out
"""
    parser = argparse.ArgumentParser(description=desc,epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--task',
                       help='Task to do. Either "enroll" or "predict"',
                       required=True)

    parser.add_argument('-i', '--input',
                       help='Input Files(to predict) or Directories(to enroll)',
                       required=True)

    parser.add_argument('-m', '--model',
                       help='Model file to save(in enroll) or use(in predict)',
                       required=True)

    ret = parser.parse_args()
    return ret

m_enroll = ModelInterface()

def task_enroll(input_dirs, output_model):
    
    input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs))
    dirs = [d for d in dirs if os.path.isdir(d)]
    files = []
    if len(dirs) == 0:
        print "No valid directory found!"
        sys.exit(1)
    for d in dirs:
        label = os.path.basename(d.rstrip('/'))

        wavs = glob.glob(d + '/*.wav')
        if len(wavs) == 0:
            print "No wav file found in {0}".format(d)
            continue
        print "Label {0} has files {1}".format(label, ','.join(wavs))
        for wav in wavs:
            fs, signal = read_wav(wav)
            signal = m_enroll.filter(fs, signal)
            m_enroll.enroll(label, fs, signal)

    m_enroll.train()
    m_enroll.dump(output_model)

def task_predict(input_file, input_model):

    # FS = 8000
    m = ModelInterface.load(input_model)
    fs_noise, noise = read_wav("noise.wav")
    m.init_noise(fs_noise, noise)
    
            
    #for f in [input_files]:
    try:
        fs, signal = read_wav(input_file)
        
        print("freq " + str(fs))
        
        signal = m.filter(fs, signal)
        print("len " + str(len(signal)))
        if len(signal) < 50:
            return None
        
        print("AA")
        label = m.predict(fs, signal)
        print input_file, '->', label
        return label
    except:
        print "Unexpected error:", sys.exc_info()[0]            
    
m = None

if __name__ == '__main__':
    global args
    args = get_args()

    fs_noise, noise = read_wav("noise.wav")
    m_enroll.init_noise(fs_noise, noise)

    task = args.task
    if task == 'enroll':
        task_enroll(args.input, args.model)
    elif task == 'predict':
        task_predict(args.input, args.model)