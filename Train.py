import os
import numpy as np
from sklearn import preprocessing
import torch
from torch import nn
import torchaudio
from torchaudio import transforms
from torch.utils.data import DataLoader, random_split
from Audio import AudioDS, LoadAudio
from Model import AudioClassifier, AttBlock, PANNsLoss
from Predict import predictFramewise
from main import training



def inference (model, val_dl, le):
    correct_prediction = 0
    total_prediction = 0
    i = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
            print("i:",i)
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)
            print(inputs.shape)
            audio_files = data[2]
    
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
    
            # Get predictions
            outputs = model(inputs)
            clip_outputs = outputs["clipwise_output"]
            frame_outputs = outputs["framewise_output"]
    
            # Get the predicted class with the highest score
            _, predicted = torch.max(clip_outputs,1)
            _, labels = torch.max(labels,1)
          
            # Count of predictions that matched the target label
            correct_prediction += (predicted == labels).sum().item()
            total_prediction += predicted.shape[0]
          
            #p = predictFramewise(frame_outputs, audio_files, le, threshold=0.5)

    
    acc = correct_prediction/total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
  

def splitAudio(path,audio_id):
    data, sr = LoadAudio.load(path+"/"+audio_id+".wav")
    length = data.shape[1]//44100
    return length

#################################################################################################################

data_path = "/UrbanSound/data"
long_path = os.path.dirname('C:/Users/Sarah/Documents/Courses/Semester 2/Deep Learning for Audio and Music/Assesment/Coursework/')+data_path

le = preprocessing.LabelEncoder()
le.fit(["air_conditioner","car_horn","children_playing","dog_bark","drilling",
           "engine_idling","gun_shot","jackhammer","siren","street_music"])

meta_data = []
for entry in os.scandir(long_path):
    for file in os.scandir(entry):
        if file.name[-4:]=='.csv':
            length = splitAudio(long_path+"/"+entry.name, file.name[0:-4])
            f = open(file, 'r')
            line = f.readline()
            relative_path = (long_path+"/"+entry.name+"/"+file.name[0:-4])
            line = line.split(',')
            class_name = line[3]
            class_id = le.transform([class_name[0:-1]])
            labels = np.zeros((10), dtype="f")
            labels[class_id] = 1
            for i in range(length):
                x = [relative_path+"_"+str(i+1)] + [line[0]]+[line[1]]+[labels]           # file, onset, offset, class
                meta_data.append(x)


data = AudioDS(meta_data)


# Random split of 80:20 between training and validation
num_items = len(data)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_data, val_data = random_split(data, [num_train, num_val])


# Create training and validation data loaders
train_dl = DataLoader(train_data, batch_size=16, shuffle=True)
val_dl = DataLoader(val_data, batch_size=16, shuffle=False)

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

model.train()

# Train
num_epochs=60   # Just for demo, adjust this higher.
training(myModel, train_dl, num_epochs, le)
