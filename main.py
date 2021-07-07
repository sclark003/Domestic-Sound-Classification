import os
import numpy as np
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models
from Audio import AudioDS
from Model import AudioClassifier, Net
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import librosa.display


def plotSpect(spec, sr):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spec, x_axis='time',  
                             y_axis='mel', sr=sr,   
                             fmax=8000, ax=ax)   
    fig.colorbar(img, ax=ax, format='%+2.0f dB')   
    ax.set(title='Mel-frequency spectrogram')


def mixup_data(x, y, use_cuda=True, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def saveCheckpoint(acc, epoch, model):
    # Save checkpoint.
    print('Saving..')
    state = {
        'model': model,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt' + '_' + str(epoch+1))


def loadCheckpoint(epoch, use_cuda):
    print('==> Resuming from checkpoint..')
    if use_cuda == True:
      checkpoint = torch.load('/content/drive/MyDrive/check/ckpt' + '_' + str(epoch))
    else:
      checkpoint = torch.load('/content/drive/MyDrive/check/ckpt' + '_' + str(epoch), map_location=torch.device('cpu'))
    model = checkpoint['model']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
    print("Checkpoint best accuracy:", best_acc.detach().cpu().numpy())
    print("Checkpoint start epoch:", start_epoch)
    return model


def training(model, train_dl, num_epochs, use_cuda):
    
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                            steps_per_epoch=int(len(train_dl)),epochs=num_epochs)
    # Repeat for each epoch
    for epoch in range(num_epochs):

        train_loss = 0
        correct = 0
        total = 0     
        
        # Repeat for each batch in the training set
        for i, (inputs, targets, sr) in enumerate(train_dl):     
            
            # Get the input features and target labels, and put them on the GPU
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()  
            
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, use_cuda)            
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))          
            inputs = inputs.unsqueeze(1) # add channel dimension
            
            # Get predictions
            outputs = model(inputs)
            
            # Keep stats for Loss and Accuracy
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            acc = 100.*correct/total
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        print("Epoch %d of %d with Loss: %.3f | Acc: %.3f%% (%d/%d)" 
              % (epoch + 1, num_epochs, train_loss/(i+1), acc, correct, total))

        saveCheckpoint(acc, epoch, model)

    print('Finished Training')


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


def inference(model, test_dl, use_cuda):

    pred = np.array([])
    true = np.array([])
        
    # Disable gradient updates
    with torch.no_grad():
        for i, (inputs, targets, sr) in enumerate(test_dl):    
            
            # Get the input features and target labels, and put them on the GPU
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()  
            
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
         
            inputs = inputs.unsqueeze(1) # add channel dimension
            inputs = inputs.double()
            
            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, predicted = torch.max(outputs.data, 1)
            
            pred = np.append(pred, predicted.detach().cpu().numpy())
            true = np.append(true, targets.detach().cpu().numpy())
            
        f1 = f1_score(true, pred, average='micro')
        conf = confusion_matrix(true, pred)
        return f1, conf

#################################################################################################################

if __name__ == "__main__":
    data_path = os.path.dirname('C:/Users/Sarah/Documents/Courses/Project/Dataset/')    # long path to dataset
    
    
    le = preprocessing.LabelEncoder()
    le.fit(["Door Knocking","Shower Running","Toilet Flushing","Vacuum Cleaning","Keyboard Typing",  # encode class labels as numeric id values
                "Coughing","Neutral"])
    

    train_meta = metaData(data_path,le)                 # get train data
    #train_meta = train_meta[0:2] + train_meta[308:310]
    test_meta = metaData(data_path, le, train=False)    # get test data
    
    train_data = AudioDS(train_meta)                    # create train dataset
    test_data = AudioDS(test_meta)                      # create test dataset
    
    # Create training and validation data loaders
    train_dl = DataLoader(train_data, batch_size=30, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=16, shuffle=False)


    # Create the model and put it on the GPU if available
    #myModel = AudioClassifier()
    v = models.resnet50(pretrained=False)
    myModel = Net(v)      # 7 classes
    
    if torch.cuda.is_available():
        device = "cuda:0"
        use_cuda = True
    else:
        device = "cpu"
        use_cuda = False
   
    #myModel = myModel.double()
    myModel = myModel.to(device, dtype=torch.double)
    next(myModel.parameters()).device      # Check that it is on Cuda
    
    
    # Train
    num_epochs=2   # Just for demo, adjust this higher.
    #training(myModel, train_dl, num_epochs, use_cuda)
    
    
    # Run inference on trained model with the validation set
    myModel = loadCheckpoint(12, use_cuda)
    
    f1, cm = inference(myModel, test_dl, use_cuda)
    print("Validation Set F1 score:",f1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot() 