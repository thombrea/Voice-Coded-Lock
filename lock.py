import numpy as np
import pyaudio
import wave
import process
import pickle

import RPi.GPIO as GPIO
import time

def main():

    ### Audio recording params
    chunk = 1024 # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 16000  # Record at 16000 samples per second
    seconds = 3
    thresh = 1e-5

    ### GPIO setup
    # Pins
    processing = 22
    unlock = 24
    lock = 26

    # Setup
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(processing, GPIO.OUT)
    GPIO.setup(unlock, GPIO.OUT)
    GPIO.setup(lock, GPIO.OUT)

    ### Open models
    # HMM
    with open('models/lambda3.pickle', 'rb') as handle:
        Lambda = pickle.load(handle)
    
    # SVM
    with open('models/clf3.pickle', 'rb') as handle:
        clf = pickle.load(handle)

    ### First call of time.time since it takes longer
    start_time = time.time()

    try:
        while(True):

            # Reset GPIO pins
            GPIO.output(processing, GPIO.HIGH)
            GPIO.output(unlock, GPIO.HIGH)
            GPIO.output(lock, GPIO.HIGH)
            print("Init")
            time.sleep(2)

            ### Open audio stream
            p = pyaudio.PyAudio()  # Create an interface to PortAudio
            stream = p.open(format=sample_format,
                            channels=channels,
                            rate=fs,
                            frames_per_buffer=chunk,
                            input=True)

            
            frames = []  # Initialize array to store frames
            npframes = np.array([])
            data = stream.read(chunk*3)

            print('Recording')
            ### Record audio
            while(True):
                data = stream.read(chunk)
                numpydata = (np.frombuffer(data,dtype=np.int16) / 32767).astype(np.float32)

                if np.mean(numpydata**2) > thresh:
                    print("Processisng")
                    GPIO.output(lock, GPIO.HIGH)
                    GPIO.output(processing, GPIO.LOW)

                    print("Threshold reached")
                    frames.append(data)
                    npframes = np.concatenate((npframes,numpydata))

                    for i in range(0, int(fs / chunk * seconds)):
                        data = stream.read(chunk)
                        numpydata = np.frombuffer(data,dtype=np.int16)
                        frames.append(data)
                        npframes = np.concatenate((npframes,numpydata))
                    break

            # Stop and close the stream 
            stream.stop_stream()
            stream.close()

            # Terminate the PortAudio interface
            p.terminate()

            print('Finished recording, length of recording = {} seconds'.format(len(npframes)/fs))

            ### Recognize
            print("Recognizing")
            
            start_time = time.time()
            X = {}
            wav = process.process_audio(npframes/32767, thresh=5e-2)

            print("Extracted length = {} seconds".format(len(wav)/16000))

            if len(wav) < chunk:
                GPIO.output(lock, GPIO.LOW)
                print("TOO QUIET, STAY LOCKED")
                time.sleep(2)
                continue

            X[0] = process.mfcc(wav)
            pred = process.predict(X,Lambda,clf,0)
            print("Recognition time = {} seconds".format(time.time()-start_time))
            
            GPIO.output(processing, GPIO.HIGH)
            if pred[0] == 0:
                GPIO.output(unlock, GPIO.LOW) # unlock door
                print("UNLOCK")
                time.sleep(1.5) # hold unlock
                GPIO.output(unlock, GPIO.HIGH) # reset unlock
                time.sleep(10) # delay for opening door
            else:
                GPIO.output(lock, GPIO.LOW)
                print("STAY LOCKED")
                time.sleep(2)
            
            
            print("Done")
            
    except KeyboardInterrupt:
        pass

    print("Exiting")
    GPIO.cleanup()

if __name__ == "__main__":
    main()