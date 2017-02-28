import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os

########################
## JRF
## VideoRecorder and AudioRecorder are two classes based on openCV and pyaudio, respectively. 
## By using multithreading these two classes allow to record simultaneously video and audio.
## ffmpeg is used for muxing the two signals
## A timer loop is used to control the frame rate of the video recording. This timer as well as
## the final encoding rate can be adjusted according to camera capabilities
##

########################
## Usage:
## 
## numpy, PyAudio and Wave need to be installed
## install openCV, make sure the file cv2.pyd is located in the same folder as the other libraries
## install ffmpeg and make sure the ffmpeg .exe is in the working directory
##
## 
## start_AVrecording(filename) # function to start the recording
## stop_AVrecording(filename)  # "" ... to stop it
##
##
########################

class AudioRecorder():
    
    inner_thread = None
    stopRecordingTime = 0

    # Audio class based on pyAudio and Wave
    def __init__(self):
        
        print("new AudioRecorder")
        self.recording = False
        self.rate = 44100
        self.frames_per_buffer = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        self.audio_filename = "temp_audio.wav"

        print("new AudioRecorder")
        self.audio = pyaudio.PyAudio()

        # PER DEBUG
        p = self.audio  
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
                if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    print "Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name')
        # FINE PER DEBUG
        
        self.open_stream()
        
        #


    def time_expired(self):
        return time.time() >= self.stopRecordingTime

    # Audio starts being recorded
    def record(self):

        print("audio thread recording...")
        self.stream.start_stream()
        while(self.recording == True and not self.time_expired()):
            data = self.stream.read(self.frames_per_buffer) 
            self.audio_frames.append(data)
        print("audio thread recording completed.")
        
        self.stream.stop_stream()
        #self.stream.close()
        #self.audio.terminate()
        print("audio closed")
           
        print("Saving file to {} ...".format(self.audio_filename))       
        waveFile = wave.open(self.audio_filename, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.audio.get_sample_size(self.format))
        waveFile.setframerate(self.rate)
        waveFile.writeframes(b''.join(self.audio_frames))
        waveFile.close()
        print("Save file done")             
        
        self.recording = False
        print("audio thread finished.")

    # Finishes the audio recording therefore the thread too    
    def stop(self):

        print("stop called")
        if self.recording==True:
            self.recording = False
            self.inner_thread.join()

    def open_stream(self):
        self.stream = self.audio.open(input_device_index = 3,
                                      format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []
    
    # Launches the audio recording function using a thread
    def start(self, recordingLengthSeconds):

        print("start called")
        
        if self.isRecording():
            print("already recording")
            return None
        
        self.recording = True
        self.inner_thread = threading.Thread(target=self.record)
        now = time.time()
        self.stopRecordingTime = now + recordingLengthSeconds
        
        self.audio_filename = "audio" + str(now) + ".wav"
        self.inner_thread.start()
        
        return self.audio_filename

    def waitForRecordingCompletion(self):
        self.inner_thread.join()
        
    def extend(self, recordingLengthSeconds):
        self.stopRecordingTime = time.time() + recordingLengthSeconds
    
    def isRecording(self):
        return self.recording == True

def start_recording(recordingLengthSeconds):
                
    global audio_recorder
    
    if audio_recorder == None:
        audio_recorder = AudioRecorder()

    return audio_recorder.start(recordingLengthSeconds)

def stop_recording():
    
    audio_recorder.stop() 
    

# Required and wanted processing of final files
def file_manager():

    local_path = os.getcwd()

    if os.path.exists(str(local_path) + "/temp_audio.wav"):
        os.remove(str(local_path) + "/temp_audio.wav")
 
    
audio_recorder = None

    
if __name__== "__main__":

    file_manager()
    
    start_recording(3)
    
    time.sleep(5)
    
    start_recording(4)
    
    #stop_AVrecording(5)

    #time.sleep(10)
    
    print "Done"



