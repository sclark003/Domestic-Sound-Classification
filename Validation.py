import os
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, random_split
from Audio import AudioDS, ClipAudioDS
from Model import AudioClassifier
from main import inference, clip_inference
import numpy as np
from sklearn.metrics import f1_score



def compareOnsets(onset_offsets, e_list):
    y_true = np.array([])
    y_pred = np.array([])
    for e in e_list:
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
            f = open(file, 'r')
            line = f.readline()
            #for line in f:
            relative_path = (long_path+"/"+entry.name+"/"+file.name[0:-4])
            line = line.split(',')
            class_name = line[3]
            class_id = le.transform([class_name[0:-1]])
            labels = np.zeros((10), dtype="f")
            labels[class_id] = 1
            x = [relative_path] + [line[0]]+[line[1]]+[labels]           # file, onset, offset, class
            #x = [relative_path] + [line[0]]+[line[1]]+[class_id[0]]     # file, onset, offset, class
            meta_data.append(x)


onset_offsets = {}
for entry in os.scandir(long_path):
    for file in os.scandir(entry):
        if file.name[-4:]=='.csv':
            f = open(file, 'r')
            onsets_mask = np.zeros(1320000)
            for line in f:
                line = line.split(",")
                if float(line[0])<30:
                    onset = float(line[0])*44100
                    offset = float(line[1])*44100
                    onsets_mask[int(onset):int(offset)+1] = 1
            onset_offsets[file.name[0:-4]] = onsets_mask


data = ClipAudioDS(meta_data)

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

model.eval()


cm = clip_inference (model, val_dl, le)


###################################################################################################################
frame_meta_data = []
audio_dict = {}
for entry in os.scandir(long_path):
    for file in os.scandir(entry):
        if file.name[-4:]=='.csv':
            #length = splitAudio(long_path+"/"+entry.name, file.name[0:-4])
            f = open(file, 'r')
            line = f.readline()
            relative_path = (long_path+"/"+entry.name+"/"+file.name[0:-4])
            line = line.split(',')
            class_name = line[3]
            class_id = le.transform([class_name[0:-1]])
            labels = np.zeros((10), dtype="f")
            labels[class_id] = 1
            audio_dict[file.name[0:-4]]=class_id
            for i in range(2579):
                x = [relative_path+"_"+str(i+1)] + [line[0]]+[line[1]]+[labels]           # file, onset, offset, class
                frame_meta_data.append(x)


onset_offsets = {}
for entry in os.scandir(long_path):
    for file in os.scandir(entry):
        if file.name[-4:]=='.csv':
            f = open(file, 'r')
            onsets_mask = np.zeros(1320000)
            for line in f:
                line = line.split(",")
                if float(line[0])<30:
                    onset = float(line[0])*44100
                    offset = float(line[1])*44100
                    onsets_mask[int(onset):int(offset)+1] = 1
            onset_offsets[file.name[0:-4]] = onsets_mask


frame_data = AudioDS(frame_meta_data)

0.808595328275957
# Random split of 80:20 between training and validation
num_items = len(frame_meta_data)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
#train_data, val_data = random_split(data, [num_train, num_val])
val_data1 = frame_meta_data[0:780]
val_data2 = frame_meta_data[3900:4680]
val_data3 = frame_meta_data[7800:8580]
val_data4 = frame_meta_data[11700:12480]
val_data5 = frame_meta_data[15600:16380]
val_data6 = frame_meta_data[19500:20280]
val_data7 = frame_meta_data[23400:24180]
val_data8 = frame_meta_data[27300:28080]
val_data9 = frame_meta_data[31200:31980]
val_data10 = frame_meta_data[35100:35880]


        
        


v = val_data1 + val_data2 + val_data3 + val_data4 + val_data5#val_data6 + val_data7 + val_data8 + val_data9 + val_data10 #val_data1 + val_data2 + val_data3 + val_data4 + val_data5 + val_data6 + val_data7 + val_data8 + val_data9 + val_data10
frame_val_data = AudioDS(v)

# Create training and validation data loaders
#train_dl = DataLoader(train_data, batch_size=30, shuffle=False)
frame_val_dl = DataLoader(frame_val_data, batch_size=30, shuffle=False)


# Run inference on trained model with the validation set
e = inference(model, frame_val_dl,le, audio_dict)
compareOnsets(onset_offsets, e)

















