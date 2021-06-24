import torch
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset


class LoadAudio():
        
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
            #pad_begin_len = random.randint(0, max_len - data_len)    # Length of padding to add at the beginning of the signal
            #pad_end_len = max_len - data_len - pad_begin_len         # Length of padding to add at the beginning of the signal    
            #pad_begin = torch.zeros((num_rows, pad_begin_len))       # Pad beginning with 0s
            #pad_end = torch.zeros((num_rows, pad_end_len))           # Pad end with 0s   
            #data = torch.cat((pad_begin, data, pad_end), 1)          # Concatenate data with padding 
            
            pad_end_len = max_len - data_len         # Length of padding to add at the beginning of the signal  
            pad_end = torch.zeros((num_rows, pad_end_len))           # Pad end with 0s 
            data = torch.cat((data, pad_end), 1)          # Concatenate data with padding          
        return (data, sr)


    # create mel spectrograms
    def spectrogram(audio, n_mels=64, n_fft=1024, hop_len=None):
        data,sr = audio
        top_db = 80
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(data)  # [channel, n_mels, time]
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec


class AudioDS(Dataset):
    def __init__(self, labels):
        self.labels = labels
        self.max_ms = 4000   # 4 secs
        self.sr = 44100
    
            
    def __len__(self):
        return len(self.labels)    
        

    def __getitem__(self, idx):

        relative_path = self.labels[idx][0]                                         # audio file location
        class_id = self.labels[idx][1]                                              # class label

        audio = LoadAudio.load(relative_path)                                       # load audio file
        audio = LoadAudio.resample(audio, self.sr)                                  # resample audio
        audio = LoadAudio.mono(audio)                                             # make audio stereo
        audio = LoadAudio.resize(audio, self.max_ms)                                # resize audio 

        sgram = LoadAudio.spectrogram(audio, n_mels=64, n_fft=1024, hop_len=None)   # create spectrogram 
        
        audio_file, sr = audio
        #return sgram, class_id, audio
        return class_id, audio_file



class TestDS(Dataset):
    def __init__(self, labels):
        self.labels = labels
        self.max_ms = 1000   # 30 secs
        self.sr = 44100
    
            
    def __len__(self):
        return len(self.labels)    
        

    def __getitem__(self, idx):
        relative_path = self.labels[idx][0]
        relative_path = relative_path.split("/")
        
        audio_id = relative_path[-1]

        audio_file = audio_id.split("_")
        id_num = audio_file[1]
        audio_file = audio_file[0]+".wav"
 
        relative_path = 'C:/Users/Sarah/Documents/Courses/Semester 2/Deep Learning for Audio and Music/Assesment/Coursework/Test_file/'+audio_file

        audio = LoadAudio.load(relative_path)                                       # load audio file
        audio = LoadAudio.resample(audio, self.sr)                                  # resample audio
        audio,sr = LoadAudio.mono(audio)                                            # make audio mono
        audio = audio.detach().numpy()
        audio = audio[0,:]
        length = len(audio)
        x = int(id_num)*(self.sr)
        if length-x < 44100:
            audio = audio[x:]
        else:
            audio = audio[x:x+self.sr]  
        audio = torch.from_numpy(audio)
        audio = audio.unsqueeze(0)
        #audio = LoadAudio.stereo(audio)                                            # make audio stereo
        audio = LoadAudio.resize(audio, sr, self.max_ms)                            # resize audio
        sgram = LoadAudio.spectrogram(audio, n_mels=64, n_fft=1024, hop_len=None)   # create spectrogram 
        return sgram, audio_id











        