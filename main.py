import os
import numpy as np
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader
from Audio import AudioDS
from Model import AudioClassifier
from sklearn.metrics import f1_score
from Predict import predictFramewise, postProcess
from sklearn.metrics import confusion_matrix


def training(model, train_dl, num_epochs, le):
  # Loss Function, Optimizer and Scheduler
  criterion = nn.CrossEntropyLoss()
  #criterion = PANNsLoss()

  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')
  # Repeat for each epoch
  for epoch in range(num_epochs):
    predictions_list = np.array([])
    labels_list = np.array([])
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0
    PATH = "check/model"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Repeat for each batch in the training set
    for i, data in enumerate(train_dl):
        print(i)
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = data[0].to(device), data[1].to(device)
        audio_files = data[2]
        
        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # Get predictions
        outputs = model(inputs)
        clip_outputs = outputs["clipwise_output"]
        frame_outputs = outputs["framewise_output"]
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()
    
        # Get the predicted class with the highest score
        _, predicted = torch.max(clip_outputs,1)
        _, labels = torch.max(labels,1)
          
        # Count of predictions that matched the target label
        correct_prediction += (predicted == labels).sum().item()
        total_prediction += predicted.shape[0]
    
        predicted = predicted.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        predictions_list = np.append(predictions_list,predicted)
        labels_list = np.append(labels_list,labels)
    
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
    class_ids = np.array([0,1,2,3,4,5,6,7,8,9])
    print("A:", f1_score(labels_list, predictions_list, labels=class_ids, average='micro'))
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            }, PATH+"_"+str(epoch)+".pt")

  print('Finished Training')


def clip_inference (model, val_dl, le):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictions_list = np.array([])
    labels_list = np.array([])
    e_list = []
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for i,data in enumerate(val_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)
            audio_files = data[2]
    
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
    
            # Get predictions
            outputs = model(inputs)
            clip_outputs = outputs["clipwise_output"]
            #frame_outputs = outputs["framewise_output"]
    
            # Get the predicted class with the highest score
            _, predicted = torch.max(clip_outputs,1)
            _, labels = torch.max(labels,1)
    
            predicted = predicted.detach().cpu().numpy()
            predictions_list = np.append(predictions_list,predicted)
            labels_list = np.append(labels_list,labels)
    
    #acc = correct_prediction/total_prediction
    #print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
    class_ids = np.array([0,1,2,3,4,5,6,7,8,9])
    conf = confusion_matrix(labels_list, predictions_list)
    print("F1:", f1_score(labels_list, predictions_list, labels=class_ids, average='micro'))
    return conf


def inference (model, val_dl, le, audio_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictions_list = np.array([])
    labels_list = np.array([])
    e_list = []
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for i,data in enumerate(val_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)
            audio_files = data[2]
    
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
    
            # Get predictions
            outputs = model(inputs)
            clip_outputs = outputs["clipwise_output"]
            frame_outputs = outputs["framewise_output"]
            
            e = inferencePrediction(outputs, data[2], le)
            e_list.append(e)
    
            # Get the predicted class with the highest score
            _, predicted = torch.max(clip_outputs,1)
            predicted_np = predicted.detach().cpu().numpy()
            predicted_np = np.bincount(predicted_np).argmax()
            _, labels = torch.max(labels,1)
         
            # Count of predictions that matched the target label
            #correct_prediction += (predicted == labels).sum().item()
            #total_prediction += predicted.shape[0]
            
            x =audio_files[0]
            labels = audio_dict[x[0:-4]]
            labels = labels[0]
    
            predicted = predicted.detach().cpu().numpy()
            predictions_list = np.append(predictions_list,predicted_np)
            labels_list = np.append(labels_list,labels)
    
    #acc = correct_prediction/total_prediction
    #print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
    class_ids = np.array([0,1,2,3,4,5,6,7,8,9])
    print("F1:", f1_score(labels_list, predictions_list, labels=class_ids, average='micro'))
    return e_list
  

def inferencePrediction(outputs, audio_names, le):
    clip_outputs = outputs["clipwise_output"]      # seperate clipwise outputs
    _, predicted = torch.max(clip_outputs,1)       # find max value for clipwise prediction
    e, class_id = predictFramewise(predicted, audio_names, le)
    e = postProcess(e)
    return e

# get data from dataset
def metaData(path,enc,train=True):
    meta_data = []
    if train==True:
        folder = "/train/"    # get train data
    else: 
        folder = "/eval/"     # get test data
        
    for entry in os.scandir(data_path):                 # for each folder corresponding to a class in dataset
        file_path = data_path+"/"+entry.name+folder     # set file path location
        for file in os.scandir(file_path):              # for each sample in class folder
            class_id = enc.transform([entry.name])      # get class numeric id according to label encoder
            relative_path = file_path+file.name         # get path location of data sample for loading audio
            x = [relative_path,class_id[0]]             # save class id and path
            meta_data.append(x)                         # append to meta data list
            
    return meta_data


#################################################################################################################

if __name__ == "__main__":
    data_path = os.path.dirname('C:/Users/Sarah/Documents/Courses/Project/Dataset/')    # long path to dataset
    
    
    le = preprocessing.LabelEncoder()
    le.fit(["Door Knocking","Shower Running","Toilet Flushing","Vacuum Cleaning","Keyboard Typing",  # encode class labels as numeric id values
                "Coughing","Neutral"])
    
    train_meta = metaData(data_path,le)                 # get train data
    test_meta = metaData(data_path, le, train=False)    # get test data
    
    train_data = AudioDS(train_meta)                    # create train dataset
    test_data = AudioDS(test_meta)                      # create test dataset
    
    # Create training and validation data loaders
    train_dl = DataLoader(train_data, batch_size=30, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=16, shuffle=False)
    
    for i, data in enumerate(train_dl):
        x = 1

    # # Create the model and put it on the GPU if available
    # myModel = AudioClassifier()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # myModel = myModel.to(device)
    # # Check that it is on Cuda
    # next(myModel.parameters()).device
    
    
    # # Train
    # num_epochs=2   # Just for demo, adjust this higher.
    # training(myModel, train_dl, num_epochs, le)
    
    # # Run inference on trained model with the validation set
    # #inference(myModel, val_dl,le)