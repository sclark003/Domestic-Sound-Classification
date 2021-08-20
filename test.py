"""
Run test on live audio input
"""


import pyaudio
import wave
import numpy as np
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models
from Audio import LoadAudio
from Model import AudioClassifier, Net
from MobileNetV2 import MobileNetV2Net
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler



def loadCheckpoint(epoch, model_type, use_cuda):
    """
    Inputs:
        epoch = The epoch of the model checkpoint you want to load
        use_coda = (Bool) True if using cuda, False if using CPU
    
    Returns:
        model = Model loaded from checkpoint
        start_epoch = The epoch of the loaded model
        train_hist = Training history statistics
    """
    print('==> Resuming from checkpoint..')
    if use_cuda == True:
        checkpoint = torch.load('checkpoint/' + model_type + '/ckpt' + '_' + str(epoch), map_location=torch.device('cuda:0'))   # load checkpoint on GPU
        #checkpoint = torch.load('/content/drive/MyDrive/check/ckpt' + '_' + str(epoch), map_location=torch.device('cuda:0'))
    else:
        checkpoint = torch.load('checkpoint/' + model_type + '/ckpt' + '_' + str(epoch), map_location=torch.device('cpu'))   # load checkpoint on CPU
        #checkpoint = torch.load('/content/drive/MyDrive/check/ckpt' + '_' + str(epoch), map_location=torch.device('cpu'))

    model = checkpoint['model']                                                # load checkpoint model        
    best_acc = checkpoint['acc']                                               # load best accuracy of that epoch from checkpoint
    start_epoch = checkpoint['epoch'] + 1                                      # get epoch number of saved checkpoint
    rng_state = checkpoint['rng_state']                                        # checkpoint rng state 
    rng_state = rng_state.type(torch.ByteTensor)                               # convert to Byte Tensor
    torch.set_rng_state(rng_state)                                     
    train_hist = checkpoint['train_hist']                                      # load training history of model
    print("Checkpoint best accuracy:", best_acc)#.detach().cpu().numpy())      # print best accuracy 
    print("Checkpoint start epoch:", start_epoch)                              # print epoch number  

    return model, start_epoch, train_hist



def modelLoad(model_type):
    """
    Load CNN model for use on Raspberry Pi
    """
    device = "cpu"       # CPU for Raspberry Pi
    use_cuda = False
       
    myModel, start_epoch, train_hist = loadCheckpoint(30, model_type, use_cuda)
    myModel = myModel.to(device, dtype=torch.double)
    next(myModel.parameters()).device      
    return myModel



def buildSgram(data):
    """
    Create sgram audio features
    """
    audio = (data, 44100)
    sgram = LoadAudio.spectrogram(audio, n_mels=64, n_fft=1024, hop_len=None)   # create spectrogram 
    sgram = LoadAudio.hpssSpectrograms(audio,sgram)
    return sgram, audio
      

def cnn(myModel, sgram):
    """
    Input sgram features to cnn audio model
    """
    inputs = torch.tensor(sgram)   # convert audio features to tensor for input to CNN model
    inputs = inputs.unsqueeze(0)   # add extra dimension (batch size)

    # Normalize the inputs
    inputs_m, inputs_s = inputs.mean(), inputs.std()
    inputs = (inputs - inputs_m) / inputs_s
             
    inputs = inputs.double()      # convert inputs to double type
    outputs = myModel(inputs)     # Get predictions

    # Get the predicted class with the highest score
    _, predicted = torch.max(outputs.data, 1)
    print("Identified:",le.inverse_transform(predicted))



def non_cnn(sgram, audio, scaler, loaded_model):
    """
    Input features to non-cnn audio model
    """
    # Extract audio features
    X = []
    z = LoadAudio.zeroCrossingRate(audio)
    m = LoadAudio.mfcc(audio)
    c = LoadAudio.spectralCentroid(audio)
    feature_matrix=np.array([])
    # use np.hstack to stack feature arrays horizontally to create a feature matrix
    feature_matrix = np.hstack((np.mean(sgram[0]), np.mean(sgram[1]), np.mean(sgram[2]), np.mean(z), np.mean(m), np.mean(c)))
    X.append(feature_matrix)

    # Scale audio features
    X = scaler.transform(X)

    # Classify
    predicted=loaded_model.predict(X)
    print("Identified:",le.inverse_transform(predicted))



"""
Code adapted from ReSpeaker microphone Github 'record_one_channel.py': 
https://github.com/respeaker/4mics_hat/blob/master/recording_examples/record_one_channel.py
"""

# set microphone recording parameters
RESPEAKER_RATE = 44100          
RESPEAKER_CHANNELS = 2            
RESPEAKER_WIDTH = 2
RESPEAKER_INDEX = 2    # refer to input device id
CHUNK = 1024           
RECORD_SECONDS = 1#4
WAVE_OUTPUT_FILENAME = "output_one_channel.wav"

#model_type = "cnn14" 
#model_type = "mobilenet"
model_type = "resnet"
myModel = modelLoad(model_type)                                                          # load CNN model
#sc = pickle.load(open('models/knn_scaler.pkl', 'rb'))                         # load non-CNN scaler
#loaded_model = pickle.load(open("models/knn_model.sav", 'rb'))                # load non-CNN model

# encode class labels as numeric id values
le = preprocessing.LabelEncoder()
le.fit(["Door Knocking","Shower Running","Toilet Flushing","Vacuum Cleaning","Keyboard Typing",  
                "Coughing","Neutral"])

p = pyaudio.PyAudio()
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n") # clear screen
print("______________________________________________________")
stream = p.open(                                                           # open stream
        rate=RESPEAKER_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=RESPEAKER_CHANNELS,
        input=True,
        input_device_index=RESPEAKER_INDEX,)

print("* listening... *")

frames = [] 
zz = np.array([])

for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    a = np.fromstring(data,dtype=np.int16)[0::2] # extract channel 0 data from 2 channels, if you want to extract channel 1, please change to [1::2]
    zz = np.append(zz, a.astype(float))          # append float audio data
    frames.append(a.tostring())

print("* done recording")

stream.stop_stream()                                                       # stop and close stream
stream.close()
    
# save as wav
#wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
#wf.setnchannels(1)
#wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
#wf.setframerate(RESPEAKER_RATE)
#wf.writeframes(b''.join(frames))
#wf.close()

sgram, audio = buildSgram(zz)                   # extract sgram features
cnn(myModel, sgram)                             # run cnn model
#non_cnn(sgram, audio, sc, loaded_model)        # run non-cnn model

p.terminate()
print("finished")
print("______________________________________________________")