import os
import numpy as np
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models
from Audio import AudioDS, LoadAudio
from Model import AudioClassifier, Net
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import librosa.display
import time
from MobileNetV2 import MobileNetV2Net
from collections import Counter
import pandas as pd



def plotSpect(spec, sr):
    """
    Function to plot spectrogram
    """
    fig, ax = plt.subplots()
    img = librosa.display.specshow(spec, x_axis='time',  y_axis='mel', sr=sr,   fmax=8000, ax=ax)   
    fig.colorbar(img, ax=ax, format='%+2.0f dB')   
    ax.set(title='Mel-frequency spectrogram')



def mixup_data(x, y, use_cuda=True, alpha=1.0):
    """
    Returns mixed inputs, pairs of targets, and lambda
    Source: https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    """
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
    """
    Returns mix-up criterion to calculate loss
    Source: https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def saveCheckpoint(acc, epoch, model, train_hist):
    """
    Save checkpoint for epoch
    """
    print('Saving..')
    state = {
        'model': model,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'train_hist': train_hist
    }
    if not os.path.isdir('checkpoint'):     # save to checkpoint directory
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt' + '_' + str(epoch+1))



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



def training(model, train_dl, num_epochs, use_cuda, train_hist = 0, start_epoch=0):
    """
    Inputs:
        model = model to train
        train_dl = Dataloader for training data
        num_epochs = Int, Number of epochs to train for
        use_cuda = Bool, True if using GPU, Flase if using CPU
        train_hist = '0' if there is no previous history, otherwise dictionary loaded from checkpoint
        start_epoch = '0' if no previous training has occurred, otherwise epoch value from checkpoint
    """
    # if no previous history, create empty dictionary
    if train_hist==0:
        train_hist = {}
        train_hist['loss'] = []
        train_hist['per_epoch_ptimes'] = []
    
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=int(len(train_dl)),epochs=num_epochs)
    start_time = time.time()
    
    # Repeat for each epoch
    for epoch in range(num_epochs):

        epoch_start_time = time.time()
        train_loss = 0
        correct = 0
        total = 0
        Loss = []
        
        # Repeat for each batch in the training set
        for i, (inputs, targets, sr) in enumerate(train_dl):     
            print(i)
            # Get the input features and target labels, and put them on the GPU
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()  
                        
            #Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            
            # apply mix-up
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, use_cuda)            
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))          
            
            # convert input type to double
            inputs = inputs.double()  

            # Get predictions
            outputs = model(inputs)
           
            # Keep stats for Loss and Accuracy
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            #loss = criterion(outputs, targets)

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            #correct += (predicted == targets).sum().item()
            
            acc = 100.*correct/total    # calculate training accuracy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # store the loss
            Loss.append(loss.item())
        
        epoch_loss = np.mean(Loss)             # mean loss for the epoch
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time   
        
        print("Epoch %d of %d with Loss: %.3f | Acc: %.3f%% (%d/%d) | Epoch Time: %.3f" 
              % (epoch + 1, num_epochs, train_loss/(i+1), acc, correct, total, per_epoch_ptime))
        
        # record the loss and time for every epoch
        train_hist['loss'].append(epoch_loss)
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

        saveCheckpoint(acc, epoch+start_epoch, model, train_hist)

    end_time = time.time()
    total_ptime = end_time - start_time
    print('Finished Training in Time: %.3f' % (total_ptime))
    #return train_hist



def metaData(path,enc,train=True):
    """
    Input: 
        data_path = path to audio dataset
        enc =label encoder used to convert class labels to numbers
        train = boolean that is true for training data, and false for testing data
    
    Returns:
        file_names = list of paths to each audio file in database
        class_ids = list of target classes corresponding to file_names list
    """
    meta_data = []
    classes = []
    if train==True:
        folder = "/train/"    # get train data
    else: 
        folder = "/eval/"     # get test data
        
    for entry in os.scandir(data_path):                 # for each folder corresponding to a class in dataset
        file_path = data_path+"/"+entry.name+folder     # set file path location
        for file in os.scandir(file_path):              # for each sample in class folder
            if file.name[-4:]!=".wav":
              pass
            else:
              class_id = enc.transform([entry.name])      # get class numeric id according to label encoder
              relative_path = file_path+file.name         # get path location of data sample for loading audio
              x = [relative_path,class_id[0]]             # save class id and path
              meta_data.append(x)                         # append to meta data list
              classes.append(class_id[0])
    return meta_data, classes



def inference(model, test_dl, use_cuda):
    """
    Inputs:
        model = model to train
        test_dl = Dataloader for test data
        use_cuda = Bool, True if using GPU, Flase if using CPU
    """
    pred = np.array([])
    true = np.array([])
        
    # Disable gradient updates
    with torch.no_grad():
        for i, (inputs, targets, sr) in enumerate(test_dl):    
            print(i)
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
            
        # f1 accuracy score and confusion matrix
        f1 = f1_score(true, pred, average='micro')
        conf = confusion_matrix(true, pred)
        return f1, conf

  

def test():
    """
    Function to load svm model and predict classes for test wav files in 'test wavs' folder
    """
    le = preprocessing.LabelEncoder()
    le.fit(["Door Knocking","Shower Running","Toilet Flushing","Vacuum Cleaning","Keyboard Typing",  # encode class labels as numeric id values
                "Coughing","Neutral"])
    
    if torch.cuda.is_available():
        device = "cuda:0"
        use_cuda = True
    else:
        device = "cpu"
        use_cuda = False
    
    myModel, start_epoch, train_hist = loadCheckpoint(31, use_cuda)
    
    #myModel = myModel.double()
    myModel = myModel.to(device, dtype=torch.double)
    next(myModel.parameters()).device      # Check that it is on Cuda
  
    file_names = []
    class_ids = []
    max_s = 1
    sr = 44100   
    for entry in os.scandir("test wavs/"):           # for each folder corresponding to a class in dataset
        class_id = entry.name                        # get class numeric id according to label encoder
        relative_path = "test wavs/"+entry.name      # get path location of data sample for loading audio
        file_names.append(relative_path)             # append to list
        class_ids.append(class_id)

    max_s = 1
    sr = 44100
    X_test = []  
    for i in range(len(file_names)):
        audio = LoadAudio.load(file_names[i])                                       # load audio file
        audio = LoadAudio.resample(audio, sr)                                       # resample audio
        audio = LoadAudio.mono(audio)                                               # make audio stereo
        audio = LoadAudio.resize(audio, max_s)                                      # resize audio 
        sgram = LoadAudio.spectrogram(audio, n_mels=128, n_fft=1024, hop_len=None)  # create spectrogram 
        sgram = LoadAudio.hpssSpectrograms(audio,sgram)
        sgram_tensor = torch.tensor(sgram)
        X_test.append(sgram_tensor)

    pred = np.array([])
    for i in range(len(X_test)):
        inputs = X_test[i]
        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s
        inputs = inputs.unsqueeze(0)
        inputs = inputs.double()
        
        # Get predictions
        outputs = myModel(inputs)

        # Get the predicted class with the highest score
        _, predicted = torch.max(outputs.data, 1)
            
        pred = np.append(pred, le.inverse_transform(predicted.detach().cpu().numpy()))
    

    df = pd.DataFrame(pred, columns=["Predicted"])                             # save predictions as a datafram column
    df['True'] = class_ids                                                     # save true class as a datafram column
    print("\nPredicted:", df)



if __name__ == "__main__":  
    data_path = os.path.dirname('F:/Copied documents/Courses/Project/Code/Dataset_z/')    # long path to dataset
    
    
    le = preprocessing.LabelEncoder()
    le.fit(["Door Knocking","Shower Running","Toilet Flushing","Vacuum Cleaning","Keyboard Typing",  # encode class labels as numeric id values
                "Coughing","Neutral"])
    

    train_meta, y_train = metaData(data_path,le)                 # get train data
    train_meta = train_meta[1108:1110] + train_meta[308:310]
    y_train = y_train[1108:1110] + y_train[308:310]
    test_meta, y_test = metaData(data_path, le, train=False)    # get test data
    
    train_data = AudioDS(train_meta)                            # create train dataset
    test_data = AudioDS(test_meta, train=False)                 # create test dataset
    
    # Create sampler to address class imbalance
    count=Counter(y_train)
    class_count=np.array([count[0],count[1],count[2],count[3],count[4],count[5],count[6]])
    weight=1./class_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight=torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    
    # Create training and validation data loaders
    train_dl = DataLoader(train_data, batch_size=30, sampler = sampler)
    test_dl = DataLoader(test_data, batch_size=16, shuffle=False)


    # Create the model and put it on the GPU if available
    myModel = AudioClassifier()                                    # CNN-14 model
    model_type = 'cnn14'
    
    #v = models.resnet50(pretrained=False)                         # ResNet model
    #myModel = Net(v)      # 7 classes
    #model_type = 'resnet'
    
    #myModel = MobileNetV2Net(3, 7)                                # MobileNetV2 model
    #model_type = 'mobilenet'
    
    if torch.cuda.is_available():
        device = "cuda:0"
        use_cuda = True
    else:
        device = "cpu"
        use_cuda = False
   
    myModel = myModel.double()
    myModel = myModel.to(device, dtype=torch.double)
    next(myModel.parameters()).device      # Check that it is on Cuda
    
    # Train
    num_epochs= 50   # Just for demo, adjust this higher.
    training(myModel, train_dl, num_epochs, use_cuda)                         # train from screatch 
    
    #myModel, start_epoch, train_hist = loadCheckpoint(5, model_type, use_cuda)            # load checpoint
    #training(myModel, train_dl, num_epochs, use_cuda, train_hist, start_epoch)# continue training loaded checkpoint
    
    # Run inference on trained model with the validation set
    myModel, start_epoch, train_hist = loadCheckpoint(30, model_type, use_cuda)
    
    # f1 score and confusion matrix
    f1, cm = inference(myModel, test_dl, use_cuda)
    print("Validation Set F1 score:",f1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    