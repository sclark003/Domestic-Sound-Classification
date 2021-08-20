import os
import re
import numpy as np
import pandas as pd
import scipy.io.wavfile as sw
#import python_speech_features as psf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from torch.utils.data import DataLoader
from sklearn import preprocessing
from Audio import LoadAudio
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

import torch
from Model import AudioFeatures
from sklearn import neighbors


def test(): 
    le = preprocessing.LabelEncoder()
    le.fit(["Door Knocking","Shower Running","Toilet Flushing","Vacuum Cleaning","Keyboard Typing",  # encode class labels as numeric id values
                "Coughing","Neutral"])
    file_names = []
    class_ids = []
    for entry in os.scandir("test wavs/"):                 # for each folder corresponding to a class in dataset
        class_id = entry.name      # get class numeric id according to label encoder
        relative_path = "test wavs/"+entry.name         # get path location of data sample for loading audio
        file_names.append(relative_path)            # append to list
        class_ids.append(class_id)

    
    X_test = featureExtractor(file_names)
    sc = pickle.load(open('models/knn_scaler.pkl', 'rb'))
    X_test = sc.transform(X_test)
    loaded_model = pickle.load(open("models/knn_model.sav", 'rb'))
    predicted=loaded_model.predict(X_test)
    # print("Predicted:", predicted, le.inverse_transform(predicted))
    # print("True:", class_ids)
    
    df = pd.DataFrame(le.inverse_transform(predicted), columns=["Predicted"])
    df['True'] = class_ids
    print("\nPredicted:", df)
    return df
    

# get data from dataset
def metaData(data_path,enc,train=True):
    file_names = []
    class_ids = []
    #max_data = 10
    #i = 0
    
    if train==True:
        folder = "/train/"    # get train data
    else: 
        folder = "/eval/"     # get test data
        
    for entry in os.scandir(data_path):                 # for each folder corresponding to a class in dataset
        file_path = data_path+"/"+entry.name+folder     # set file path location
        for file in os.scandir(file_path):              # for each sample in class folder
            #if i>max_data:
            #    return file_names, class_ids
            class_id = enc.transform([entry.name])      # get class numeric id according to label encoder
            relative_path = file_path+file.name         # get path location of data sample for loading audio
            file_names.append(relative_path)            # append to list
            class_ids.append(class_id)                # append to list
            #i+=1    
    return file_names, class_ids


def featureExtractor(file_names):
    max_s = 1   # 4 secs
    sr = 44100
    X = []
    #Feature Extraction
    for i in file_names:
        print(i)
        audio = LoadAudio.load(i)
        audio = LoadAudio.resample(audio, sr)                                  # resample audio
        audio = LoadAudio.mono(audio)                                               # make audio stereo
        audio = LoadAudio.resize(audio, max_s)                                 # resize audio
        audio = LoadAudio.resize(audio, max_s)                                 # resize audio
        #audio = LoadAudio.pitchShift(audio, pitch_shift)                       # apply pitch shift
        sgram = LoadAudio.spectrogram(audio, n_mels=64, n_fft=1024, hop_len=None)   # create spectrogram 
        sgram = LoadAudio.hpssSpectrograms(audio,sgram)
        #sgram = LoadAudio.nnSpectrograms(audio,sgram)

        z = LoadAudio.zeroCrossingRate(audio)
        m = LoadAudio.mfcc(audio)
        c = LoadAudio.spectralCentroid(audio)
        
        feature_matrix=np.array([])
        # use np.hstack to stack our feature arrays horizontally to create a feature matrix
        feature_matrix = np.hstack((np.mean(sgram[0]), np.mean(sgram[1]), np.mean(sgram[2]), np.mean(z), np.mean(m), np.mean(c)))
        
        X.append(feature_matrix)
        #X.append(np.mean(sgram))
        
    return np.array(X)



def loadData():
    X_train = np.load("models/X_train.npy")
    y_train = np.load("models/y_train.npy")
    y_test = np.load("models/y_test.npy")
    X_test = np.load("models/X_test.npy")    
    return X_train, y_train, X_test, y_test


def runAll():
    # Importing directory
    data_path = os.path.dirname('C:/Users/Sarah/Documents/Courses/Project/Code/Dataset/')    # long path to dataset
    
    
    le = preprocessing.LabelEncoder()
    le.fit(["Door Knocking","Shower Running","Toilet Flushing","Vacuum Cleaning","Keyboard Typing",  # encode class labels as numeric id values
                "Coughing","Neutral"])

    
    file_names, y_train = metaData(data_path, le)
    file_names_test, y_test = metaData(data_path, le, train=False)    
    
    X_train = featureExtractor(file_names)
    X_test = featureExtractor(file_names_test)
    print(X_train.shape)
    

    # np.save("models/X_train", X_train)
    # np.save("models/X_test", X_test)
    # np.save("models/y_test", y_test)
    # np.save("models/y_train", y_train)
    
    # X_train = X_train.reshape(-1, 1)
    # X_test = X_test.reshape(-1, 1)
    
    #Feature Scaling
    print("Scaling")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    y_train = np.ravel(y_train)
    # X_train=pd.DataFrame(X_train)
    # y_train=pd.DataFrame(y_train)
    # X_test=pd.DataFrame(X_test)
    # y_test=pd.DataFrame(y_test)


    
    #Model Building
    print("SVM")
    #SVM
    
    # grid_params = {
    #     'C': [1000, 1500, 2000],
    #     'gamma': [1, 0.1, 0.001, 0.0001],
    #     'kernel': ['linear', 'rbf']
    #     }    
    # model = GridSearchCV(svm.SVC(random_state=1), grid_params, cv=3, verbose=2)
    
    # model_fit = model.fit(X_train,y_train)
    # best_model = model_fit.best_estimator_
    # best_model.score(X_train,y_train)
    # predicted=best_model.predict(X_test)
    
    ### Best params found to be: {'C': 1500, 'gamma': 1, 'kernel': 'rbf'} ###
    
    model = svm.SVC(C = 1500, gamma = 1, kernel = 'rbf', probability=True)
    model_fit = model.fit(X_train,y_train)
    predicted=model_fit.predict(X_test)

    print("\n__________________________________________\n")
    accuracy_score(y_test,predicted)
    f1 = f1_score(y_test, predicted, average='micro')
    cm = confusion_matrix(y_test, predicted)
    print("Validation Set F1 score:",f1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot()
    
    # pickle.dump(model_fit, open('models/svm_model.sav', 'wb'))
    # pickle.dump(sc, open('models/svm_scaler.pkl', 'wb'))
    
    
def runModel(X_train, y_train, X_test, y_test,le):
    #Feature Scaling
    print("Scaling")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    y_train = np.ravel(y_train)

    

    #Model Building
    print("SVM")
    #SVM
    
    # grid_params = {
    #     'estimator__C': [1000, 1500, 2000],
    #     'estimator__gamma': [1, 0.1, 0.001, 0.0001],
    #     'estimator__kernel': ['linear', 'rbf']
    #     }    
    # model = GridSearchCV(OneVsRestClassifier(SVC()), grid_params, verbose=2)#random_state=1), grid_params, cv=3, verbose=2))
    
    # model_fit = model.fit(X_train,y_train)
    # best_model = model_fit.best_estimator_
    # best_model.score(X_train,y_train)
    # predicted=best_model.predict(X_test)
    
    ### Best params found to be: {'C': 1500, 'gamma': 1, 'kernel': 'rbf'} ###
    
    #model = SVC(C = 1500, gamma = 1, kernel = 'rbf', probability=True)
    modelknn = neighbors.KNeighborsClassifier(metric= 'manhattan', n_neighbors= 16, weights= 'distance')
    model_fit = OneVsRestClassifier(modelknn).fit(X_train,y_train)
    #model_fit = OneVsRestClassifier(model).fit(X_train,y_train)
    predicted=model_fit.predict(X_test)

    print("\n__________________________________________\n")
    accuracy_score(y_test,predicted)
    f1 = f1_score(y_test, predicted, average='micro')
    cm = confusion_matrix(y_test, predicted)
    print("Validation Set F1 score:",f1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot()
    
    pickle.dump(model_fit, open('models/ovr_knn_model.sav', 'wb'))
    pickle.dump(sc, open('models/ovr_knn_scaler.pkl', 'wb'))
 

def findAccuracy(X_test, y_test, le):
    sc = pickle.load(open('models/ovr_knn_scaler.pkl', 'rb'))
    X_test = sc.transform(X_test)
    loaded_model = pickle.load(open("models/ovr_knn_model.sav", 'rb'))
    predicted=loaded_model.predict(X_test)
    f1 = f1_score(y_test, predicted, average='micro')
    cm = confusion_matrix(y_test, predicted)
    print("Validation Set F1 score:",f1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot()


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = loadData()
    le = preprocessing.LabelEncoder()
    le.fit(["Door Knocking","Shower Running","Toilet Flushing","Vacuum Cleaning","Keyboard Typing",  # encode class labels as numeric id values
                "Coughing","Neutral"])
    #runModel(X_train, y_train, X_test, y_test, le)
    #runAll()
    #findAccuracy(X_test, y_test, le)
    df = test()

    


# # save the model to disk
# pickle.dump(best_model, open('models/svm_model.sav', 'wb'))
# pickle.dump(sc, open('models/svm_scaler.pkl', 'wb'))
 
# # some time later...
 
# # load the model from disk
# loaded_model = pickle.load(open("SVM_model.sav", 'rb'))
# result = loaded_model.score(X_test, y_test)
# print(result)