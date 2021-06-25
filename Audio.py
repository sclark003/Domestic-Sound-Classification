import torch
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset
import random
import librosa
import numpy as np


class LoadAudio():
        
    # load audio data
    def load(audio_file):
        data, sr = librosa.load(audio_file,sr=None,mono=False)
        return (data, sr)
    
    
    # resample audio to standard rate so all arrays have same dimensions
    def resample(audio, rate):
        data, sr = audio   
        if (rate == sr):                  # ignore if audio is already at correct sampling rate
             return audio    
        else:
            data2 = librosa.resample(data, sr, rate)                           # merge channels  
        return (data2, rate)
    
    
    # convert all audio to mono
    def mono(audio):
        data, sr = audio
        if (data.shape[0] == 1):                                   # ignore if audio is already stereo
            return audio
        else:
            data2 = librosa.to_mono(data)                          # Convert to mono by averaging samples across channels   
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
            
        return (data, sr)

    # shift pitch of audio data by random amount    
    def pitchShift(audio,shift_int):
        data, sr = audio
        n = random.randint(0-shift_int,shift_int)
        shifted_data = librosa.effects.pitch_shift(data,sr,n)
        return (shifted_data, sr)

    # create mel spectrograms
    def spectrogram(audio, n_mels=64, n_fft=1024, hop_len=None):
        data,sr = audio
        spec = librosa.feature.melspectrogram(data, sr)              # [channel, n_mels, time]
        spec_dB = librosa.core.power_to_db(spec, ref=np.max)         # convert to decibels                                        
        return spec_dB
    
    # # spectogram augmentation
    # def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    #     _, n_mels, n_steps = spec.shape
    #     mask_value = spec.mean()
    #     aug_spec = spec
    
    #     freq_mask_param = max_mask_pct * n_mels
    #     for _ in range(n_freq_masks):
    #       aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
    
    #     time_mask_param = max_mask_pct * n_steps
    #     for _ in range(n_time_masks):
    #       aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
    
    #     return aug_spec


class AudioDS(Dataset):
    def __init__(self, labels):
        self.labels = labels
        self.max_s = 4   # 4 secs
        self.sr = 44100
        self.pitch_shift = 3
    
            
    def __len__(self):
        return len(self.labels)    
        

    def __getitem__(self, idx):

        relative_path = self.labels[idx][0]                                         # audio file location
        class_id = self.labels[idx][1]                                              # class label

        audio = LoadAudio.load(relative_path)                                       # load audio file
        audio = LoadAudio.resample(audio, self.sr)                                  # resample audio
        
        audio = LoadAudio.mono(audio)                                               # make audio stereo
        audio = LoadAudio.resize(audio, self.max_s)                                # resize audio 

        audio = LoadAudio.pitchShift(audio, self.pitch_shift)

        sgram = LoadAudio.spectrogram(audio, n_mels=64, n_fft=1024, hop_len=None)   # create spectrogram 

        #aug_sgram = LoadAudio.spectro_augment(sgram)
        
        sgram_tensor = torch.tensor(sgram)
        
        audio_file, sr = audio
        return sgram_tensor, class_id, audio_file



class LoadTorchAudio():
        
    # load audio data
    def load(audio_file):
        data, sr = torchaudio.load(audio_file)
        return (data, sr)
    
    
    # resample audio to standard rate so all arrays have same dimensions
    def resample(audio, rate):
        data, sr = audio   
        if (rate ==sr):                  # ignore if audio is already at correct sampling rate
            return audio    
        num_channels = data.shape[0]                                          # find number of channels
        data2 = torchaudio.transforms.Resample(sr, rate)(data[:1,:])          # resample first audio channel
        if (num_channels > 1):
            channel2 = torchaudio.transforms.Resample(sr, rate)(data[1:,:])   # resample second channel
            data2 = torch.cat([data2, channel2])                              # merge channels  
        return (data2, rate)
    
    
    # convert all audio to stereo
    def stereo(audio):
        data, sr = audio
        if (data.shape[0] == 2):                                   # ignore if audio is already stereo
            return audio
        else:
            data2 = torch.cat([data, data])                        # Convert from mono to stereo by duplicating the first channel   
        return (data2, sr)
    
    
    # convert all audio to mono
    def mono(audio):
        data, sr = audio
        if (data.shape[0] == 1):                                   # ignore if audio is already stereo
            return audio
        else:
            data2 = data[:1, :]                        # Convert from mono to stereo by duplicating the first channel   
        return data2, sr
    
    
    # resize all audio to same length
    def resize(audio, max_ms):
        data, sr = audio
        num_rows, data_len = data.shape
        max_len = sr//1000 * max_ms    
        if (data_len > max_len):                                   # Truncate the signal to the given length     
          data = data[:,:max_len]    
        elif (data_len < max_len):
            pad_begin_len = random.randint(0, max_len - data_len)    # Length of padding to add at the beginning of the signal
            pad_end_len = max_len - data_len - pad_begin_len         # Length of padding to add at the beginning of the signal    
            
            pad_begin = torch.zeros((num_rows, pad_begin_len))       # Pad beginning with 0s
            pad_end = torch.zeros((num_rows, pad_end_len))           # Pad end with 0s   
            data = torch.cat((pad_begin, data, pad_end), 1)          # Concatenate data with padding 
            
        return (data, sr)


    # create mel spectrograms
    def spectrogram(audio, n_mels=64, n_fft=1024, hop_len=None):
        data,sr = audio
        top_db = 80
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(data)  # [channel, n_mels, time]
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)                                        # convert to decibels
        return spec
    
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
    
        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
          aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)
    
        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
          aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
    
        return aug_spec











        