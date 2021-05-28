#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 09:51:55 2021

@author: soham
"""
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
import scipy.signal as sps
import numpy as np
import wave
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def fourierTransform(data, plot=False):
    fft_out = fft(data)
    
    if plot:
        plt.figure()
        plt.plot(np.abs(fft_out))
        plt.show()
        
    return np.abs(fft_out)


def plot(audio_file):
    spf = wave.open(audio_file, "r")
    
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "Int16")
    
    if spf.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)
    
    plt.figure()
    plt.title(audio_file)
    plt.plot(signal)
    plt.show()    


def refineData(file, plot=False):
    rate, data = wav.read(file)
    num_samples = 512*20
    data = sps.resample(data, num_samples)
    data = np.split(data, 20)
    
    if plot:
        plt.figure()
        plt.plot(data[15])
        plt.show()
    
    return data
    
path = "/Users/soham/Desktop/Flock Stress Detection/"
normal_files = [path+'normal/released.wav', path+'normal/released2.wav']
stressed_files = [path+'stressed/confined.wav', path+'stressed/confined2.wav']

test_data = refineData(path+'normal/released.wav')

normal_data = []
for f in normal_files:
    for d in refineData(f):
        normal_data.append(fourierTransform(d))
    
stressed_data = []
for f in stressed_files:
    for d in refineData(f):
        stressed_data.append(fourierTransform(d))
        
data = normal_data + stressed_data 

labels = []
for i in range(len(normal_data)):
    labels.append(0)
for i in range(len(normal_data), len(data)):
    labels.append(1)

kn_model = KNeighborsClassifier(n_neighbors=3)
kn = kn_model.fit(data, labels)

nn_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
nn = nn_model.fit(data, labels)

point = 0
for p in nn.predict(test_data):
    if p == 0:
        normal = True
        print(str(point)+": normal")
    else:
        print(str(point)+": stressed")
    point += 1