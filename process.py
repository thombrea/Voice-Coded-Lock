import numpy as np
from scipy import signal
import librosa
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import hmm

def extract_audio(waveform,sample_rate=16000,window_length=0.05,thresh=1e-5):
    sample_length = int(window_length * sample_rate)
    nframes = int((len(waveform))/sample_length) + 1
    
    if len(waveform) % nframes != 0:
        pad_amt = nframes * sample_length - len(waveform)
        waveform = np.pad(waveform, (0,pad_amt))
    
    split = np.array(np.split(waveform, nframes))
    #means = np.mean(split**2,axis=1)
    std = np.std(split**2, axis=1)
    out = np.array([])
    for i in range(len(std)):
        if std[i] > thresh:
            out = np.append(out, split[i])
    
    return out

def process_audio(wav, thresh):
    wav /= np.max(wav)
    wav = signal.savgol_filter(wav,11,2) # clean background noise
    wav = extract_audio(wav,window_length=0.1,thresh=thresh) # threshold, only keep significant part of signal
    
    return wav

def mfcc(wav, fs=16000, n=15, L=30):
    mfccs = np.array(librosa.feature.mfcc(y=wav,sr=fs,lifter=L,n_mfcc=n,n_fft=1024))
    mfccs = mfccs[2:,]

    return mfccs.T

def predict(X, Lambda, clf, boost=0):
    logprob = hmm.recognize(X, Lambda)
    logprob[0][0] += boost
    print(logprob)
    
    svmfeat = np.zeros((len(logprob[0]),3))
    svmfeat[:,0] = logprob[0]
    svmfeat[:,1] = logprob[1]
    svmfeat[:,2] = np.array(logprob[0])-np.array(logprob[1])
    predictY = clf.predict(svmfeat)

    return predictY