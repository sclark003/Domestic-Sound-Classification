from sklearn import preprocessing
import numpy as np
import pandas as pd
import torch
from Audio import LoadAudio, TestDS
from Model import AudioClassifier
from Predict import predictFramewise, postProcess
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


def splitAudio(path):
    meta_data = []
    data, sr = LoadAudio.load(path)
    length = data.shape[1]//44100
    audio_file = path.split("/")
    audio_file = audio_file[-1]
    audio_file = audio_file[0:-4]
    for i in range(length):
        x = ["C:/Users/Sarah/Documents/Courses/Semester 2/Deep Learning for Audio and Music/Assesment/Coursework/Test_file/"+audio_file+"_"+str(i+1)]           # file, onset, offset, class
        meta_data.append(x)
    return meta_data

    

def makeFramePrediction(data_dl, le, class_id):
    for i,data in enumerate(data_dl):
        #data[0].shape = batch, channels, mels, frames
        #data[1] = audio names
        outputs = model(data[0])                       # get model predictions
        clip_outputs = outputs["clipwise_output"]      # seperate clipwise outputs
        _, predicted = torch.max(clip_outputs,1)       # find max value for clipwise prediction
        e, class_id = predictFramewise(predicted, data[1], le)
        e = postProcess(e)
        prediction_df = pd.DataFrame(e)
    return prediction_df, e


def preProcess(audio_file):

        #max_ms = 30000   # 30 secs
        sr = 44100
    
        audio = LoadAudio.load(audio_file)                                          # load audio file
        audio = LoadAudio.resample(audio, sr)                                       # resample audio
        audio = LoadAudio.mono(audio)                                               # make audio mono
        #audio = LoadAudio.stereo(audio)                                            # make audio stereo
        #audio = LoadAudio.resize(audio, max_ms)                                     # resize audio
        sgram = LoadAudio.spectrogram(audio, n_mels=64, n_fft=1024, hop_len=None)   # create spectrogram 
        return sgram, audio_file
    

def makeClipPrediction(data):
    sgram, audio_file = preProcess(data)         # preprocess wav file
    audio_file = audio_file.split("/")
    audio_file = audio_file[-1]
    sgram = sgram.unsqueeze(0)                   # add 'batch size' dimension for single file input
    outputs = model(sgram)                       # get model predictions
    clip_outputs = outputs["clipwise_output"]    # seperate into clipwise and framewise outputs
    #frame_outputs = outputs["framewise_output"]
    _, predicted = torch.max(clip_outputs,1)     # find max value for clipwise prediction
    
    #p, e = predictFramewise(frame_outputs, [audio_file], le, predicted.detach().cpu().numpy(), threshold=0.5) # predict framewise
    return predicted


def compareOnsets(onset_offsets, e):
    y_true = np.array([])
    y_pred = np.array([])
    onsets_mask = np.zeros(1320000)
    for event in e:
        audio_id = event['audio_id']
        true = onset_offsets[audio_id]
        onset = event['onset']*44100
        offset = event['offset']*44100
        onsets_mask[int(onset):int(offset)+1] = 1
        y_true = np.append(y_true,true)
        y_pred = np.append(y_pred,onsets_mask)
    print("F1 onsets:", f1_score(y_true, y_pred))
       




PATH = "check/model"
epoch = 4

model = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = model.to(device)
next(model.parameters()).device
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

checkpoint = torch.load(PATH+"_"+str(epoch)+".pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']



le = preprocessing.LabelEncoder()            # get class id numbers
le.fit(["air_conditioner","car_horn","children_playing","dog_bark","drilling",
        "engine_idling","gun_shot","jackhammer","siren","street_music"])

class_id = makeClipPrediction("C:/Users/Sarah/Documents/Courses/Semester 2/Deep Learning for Audio and Music/Assesment/Coursework/Test_file/Test.wav")
clip_p = le.inverse_transform(class_id)
print("Overall class prediction:", clip_p)



data = "C:/Users/Sarah/Documents/Courses/Semester 2/Deep Learning for Audio and Music/Assesment/Coursework/Test_file/Test.wav"
data_onsets = "C:/Users/Sarah/Documents/Courses/Semester 2/Deep Learning for Audio and Music/Assesment/Coursework/Test_file/Test.csv"


onset_offsets = {}
f = open(data_onsets, 'r')
onsets_mask = np.zeros(1320000)
for line in f:
    line = line.split(",")
    if float(line[0])<30:
        onset = float(line[0])*44100
        offset = float(line[1])*44100
        onsets_mask[int(onset):int(offset)+1] = 1
    onset_offsets['Test'] = onsets_mask
    
meta_data = splitAudio(data)
data = TestDS(meta_data)

# Random split of 80:20 between training and validation
num_items = len(data)
data_dl = DataLoader(data, batch_size=num_items, shuffle=False)

frame_p, e = makeFramePrediction(data_dl, le, class_id)

#print("Overall class prediction:", clip_p)
print("Onset and Offset Predictions:")
print(frame_p)

print("Onset and Offset Predictions:")
print(frame_p)