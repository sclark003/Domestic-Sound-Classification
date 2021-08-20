"""
Audio Pre-processing Classes
"""


import torch
from torch.utils.data import Dataset
import random
import librosa
import numpy as np

"""
Class to apply data pre-processing for model input
"""
class LoadAudio():
        
    # load audio data
    def load(audio_file):
        data, sr = librosa.load(audio_file,sr=None,mono=False)
        #print(data.shape)
        return (data, sr)
    
    
    # resample audio to standard rate so all arrays have same dimensions
    def resample(audio, rate):
        data, sr = audio   
        if (rate == sr):                  # ignore if audio is already at correct sampling rate
             return audio    
        else:
            data2 = librosa.resample(data, sr, rate)                           # merge channels  
        #print(data2.shape)
        return (data2, rate)
    
    
    # convert all audio to mono
    def mono(audio):
        data, sr = audio
        if (data.shape[0] == 1):                                   # ignore if audio is already stereo
            return audio
        else:
            data2 = librosa.to_mono(data)                          # Convert to mono by averaging samples across channels   
        #print(data2.shape)
        return data2, sr
    
    
    # resize all audio to same length
    def resize(audio, max_s):
        data, sr = audio
        data_len = len(data)
        max_len = sr * max_s    
        if (data_len > max_len):                                   # Truncate the signal to the given length     
            data = data[:max_len]    
        elif (data_len < max_len):
            pad_begin_len = random.randint(0, max_len - data_len)    # Length of padding to add at the beginning of the signal
            pad_end_len = max_len - data_len - pad_begin_len         # Length of padding to add at the beginning of the signal    
            
            pad_begin = np.zeros(pad_begin_len)          # Pad beginning with 0s
            pad_end = np.zeros(pad_end_len)              # Pad end with 0s   
            data = np.concatenate((pad_begin, data, pad_end))          # Concatenate data with padding 
        
        #print(data.shape)
        return (data, sr)

    
    # add random noise
    def add_noise(audio):
        data, sr = audio
        noise = np.random.randn(len(data))
        noisy_data = data + (0.0005*noise)
        return (noisy_data, sr)


    # shift pitch of audio data by random amount    
    def pitchShift(audio,shift_int):
        data, sr = audio
        n = random.randint(0-shift_int,shift_int)
        shifted_data = librosa.effects.pitch_shift(data,sr,n)
        #print(shifted_data.shape)
        return (shifted_data, sr)


    # create mel spectrograms
    def spectrogram(audio, n_mels=128, n_fft=1024, hop_len=None):
        data,sr = audio
        spec = librosa.feature.melspectrogram(data, sr)              # [channel, n_mels, time]
        spec_dB = librosa.core.power_to_db(spec, ref=np.max)         # convert to decibels                                        
        #print(spec.shape)
        return spec_dB
    
    
    # hpss spectrograms
    def hpssSpectrograms(audio,sgram):
        data, sr = audio
        sgram = np.expand_dims(sgram, axis=0)             # add extra dimension
        
        D = librosa.feature.melspectrogram(data, sr)      # create Mel spect of audio data
        H, P = librosa.decompose.hpss(D)                  # Apply HPSS to Mel spect
        H = np.expand_dims(H, axis=0)                     # add extra dimension 
        P = np.expand_dims(P, axis=0)                     # add extra dimension
        H_dB = librosa.core.power_to_db(H, ref=np.max)    # convert to decibels
        P_dB = librosa.core.power_to_db(P, ref=np.max)    # convert to decibels

        sgram = np.concatenate((sgram, H_dB), axis=0)     # combine spectrograms
        sgram = np.concatenate((sgram, P_dB), axis=0)     # combine spectrograms
        
        #print(sgram.shape)
        return sgram                               # return combined spectrogram
    
    # nn filtering
    def nnSpectrograms(audio,sgram):
        data, sr = audio
        D = librosa.feature.melspectrogram(data, sr)      # create Mel spect of audio data
        nn = librosa.decompose.hpss(D)                    # Apply nn to Mel spect
        nn_dB = librosa.core.power_to_db(nn, ref=np.max)  # convert to decibels
        sgram = np.concatenate((sgram, nn_dB), axis=0)     # combine spectrograms
        
        return sgram 
    
    
    # zero crossing rate
    def zeroCrossingRate(audio):
        data, sr = audio
        z =librosa.feature.zero_crossing_rate(data)
        z = np.expand_dims(z, axis=0)
        return z
    
    # mfcc
    def mfcc(audio):
        data, sr = audio
        m = librosa.feature.mfcc(y=data, sr=sr)
        m = np.expand_dims(m, axis=0)
        return m
    
    # spectral centroids
    def spectralCentroid(audio):
        data, sr = audio
        c = librosa.feature.spectral_centroid(data)
        c = np.expand_dims(c, axis=0)
        return c

"""
Class to load dataset for CNN models
"""
class AudioDS(Dataset):
    def __init__(self, labels, train=True):
        self.labels = labels
        self.max_s = 1       # max length of audio input = 1 sec
        self.sr = 44100      # audio input samplig rate
        self.pitch_shift = 3 # maximum pitch shift amount
        self. train = train  # train or test data (bool)
    
            
    def __len__(self):
        return len(self.labels)    
        

    def __getitem__(self, idx):

        relative_path = self.labels[idx][0]                                         # audio file location
        class_id = self.labels[idx][1]                                              # class label

        audio = LoadAudio.load(relative_path)                                       # load audio file
        audio = LoadAudio.resample(audio, self.sr)                                  # resample audio
        
        audio = LoadAudio.mono(audio)                                               # make audio stereo
        audio = LoadAudio.resize(audio, self.max_s)                                 # resize audio 

        if self.train == True:
            audio = LoadAudio.pitchShift(audio, self.pitch_shift)                    # apply pitch shift
        #audio = LoadAudio.add_noise(audio)                                          # add random noise 
        
        sgram = LoadAudio.spectrogram(audio, n_mels=128, n_fft=1024, hop_len=None)   # create spectrogram 
        sgram = LoadAudio.hpssSpectrograms(audio,sgram)                              # create HPSS spectrograms 
        
        sgram_tensor = torch.tensor(sgram)                                           # convert to tensor
        audio_file, sr = audio
        return sgram_tensor, class_id, sr

