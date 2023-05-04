import pyaudio
import wave
import numpy as np

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 16000  # Record at 16000 samples per second
seconds = 3

thresh = 1e-5


for n in range(50):
    filename = "recordings/1/" + str(250 + n) + ".wav"
    
    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input_device_index = 1,
                    input=True)

    
    frames = []  # Initialize array to store frames
    npframes = np.array([])

    data = stream.read(chunk*3)

    print('Recording ' + str(n))
    # Store data in chunks for 3 seconds
    while(True):
        data = stream.read(chunk)
        numpydata = (np.frombuffer(data,dtype=np.int16) / 32767).astype(np.float32)
        if np.mean(numpydata**2) > thresh:
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

    print('Finished recording\n')

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
