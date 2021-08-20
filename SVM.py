"""
SVM model classification
"""


import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from Audio import LoadAudio
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle



def test(): 
    """
    Function to load svm model and predict classes for test wav files in 'test wavs' folder
    """
    le = preprocessing.LabelEncoder()
    le.fit(["Door Knocking","Shower Running","Toilet Flushing","Vacuum Cleaning","Keyboard Typing",  # encode class labels as numeric id values
                "Coughing","Neutral"])
    file_names = []
    class_ids = []
    for entry in os.scandir("test wavs/"):                                                           # for each folder corresponding to a class in dataset
        class_id = entry.name                                                                        # get class numeric id according to label encoder
        relative_path = "test wavs/"+entry.name                                                      # get path location of data sample for loading audio
        file_names.append(relative_path)                                                             # append audio path to list
        class_ids.append(class_id)                                                                   # append class id to list

    
    X_test = featureExtractor(file_names)                                                            # extract features  from audio data
    sc = pickle.load(open('models/svm_scaler.pkl', 'rb'))                                            # load svm model scaler to apply correct scale
    X_test = sc.transform(X_test)                                                                    # scale features
    loaded_model = pickle.load(open("models/svm_model.sav", 'rb'))                                   # load trained svm model   
    predicted=loaded_model.predict(X_test)                                                           # make predictions
    
    df = pd.DataFrame(le.inverse_transform(predicted), columns=["Predicted"])                        # save predictions as a datafram column
    df['True'] = class_ids                                                                           # save true classes as dataframe column
    print("\nPredicted:", df)                                                                        # displat dataframe of predicted vs. true classes

    

def metaData(data_path,enc,train=True):
    """
    Input: 
        data_path = path to audio dataset
        enc =label encoder used to convert class labels to numbers
        train = boolean that is true for training data, and false for testing data
    
    Returns:
        file_names = list of paths to each audio file in database
        class_ids = list of target classes corresponding to file_names list
    """
    file_names = []
    class_ids = []
    
    if train==True:
        folder = "/train/"    # get train data
    else: 
        folder = "/eval/"     # get test data
        
    for entry in os.scandir(data_path):                 # for each folder corresponding to a class in dataset
        file_path = data_path+"/"+entry.name+folder     # set file path location
        for file in os.scandir(file_path):              # for each sample in class folder
            class_id = enc.transform([entry.name])      # get class numeric id according to label encoder
            relative_path = file_path+file.name         # get path location of data sample for loading audio
            file_names.append(relative_path)            # append path to list
            class_ids.append(class_id)                  # append class id to list  

    return file_names, class_ids



def featureExtractor(file_names):
    """
    Input:
        file_names = list of paths to each audio file in database
    
    Returns:
        numpy array of audio features for each audio file within file_names
    """
    max_s = 1   # maximum length that audio sample should be in secs
    sr = 44100  # sampling rae
    X = []       

    #Feature Extraction
    for i in file_names:
        #print(i)
        audio = LoadAudio.load(i)                                                   # load audio data
        audio = LoadAudio.resample(audio, sr)                                       # resample audio so all have the same sampling rate
        audio = LoadAudio.mono(audio)                                               # make audio stereo
        audio = LoadAudio.resize(audio, max_s)                                      # resize audio
        sgram = LoadAudio.spectrogram(audio, n_mels=64, n_fft=1024, hop_len=None)   # create mel spectrogram 
        sgram = LoadAudio.hpssSpectrograms(audio,sgram)                             # create HPSS spectrograms
                 
        z = LoadAudio.zeroCrossingRate(audio)                                       # get zero crossing rate
        m = LoadAudio.mfcc(audio)                                                   # get mfccs
        c = LoadAudio.spectralCentroid(audio)                                       # get spectral centroids
            
        feature_matrix=np.array([])                                                           
        # use np.hstack to stack feature arrays horizontally to create a feature matrix                                        
        feature_matrix = np.hstack((np.mean(sgram[0]), np.mean(sgram[1]), np.mean(sgram[2]), np.mean(z), np.mean(m), np.mean(c))) 
        X.append(feature_matrix)
        
    return np.array(X)




def loadData():
    """
    Function to load saved numpy arrays of pre-extracted features
    (These were saved to save time during testing)
    """
    X_train = np.load("models/X_train.npy")
    y_train = np.load("models/y_train.npy")
    y_test = np.load("models/y_test.npy")
    X_test = np.load("models/X_test.npy")    
    
    return X_train, y_train, X_test, y_test



def runAll():
    """
    Function to run entire process of creating and testing an SVM classifier on this dataset
    """
    # Importing directory
    data_path = os.path.dirname('F:/Copied documents/Courses/Project/Code/Dataset_z/')               # long path to dataset
    
    
    le = preprocessing.LabelEncoder()
    le.fit(["Door Knocking","Shower Running","Toilet Flushing","Vacuum Cleaning","Keyboard Typing",  # encode class labels as numeric id values
                "Coughing","Neutral"])

    
    file_names, y_train = metaData(data_path, le)                              # get training data 
    file_names_test, y_test = metaData(data_path, le, train=False)             # get testing data
    
    X_train = featureExtractor(file_names)                                     # extract features from training data
    X_test = featureExtractor(file_names_test)                                 # extract features from test data         

    # save extracted features
    # np.save("models/X_train", X_train)                                          
    # np.save("models/X_test", X_test)
    # np.save("models/y_test", y_test)
    # np.save("models/y_train", y_train)
        
    #Feature Scaling
    print("Scaling")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)            
    
    y_train = np.ravel(y_train)
    
    # Model Building
    
    # search for best parameters
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
    
    model = svm.SVC(C = 1500, gamma = 1, kernel = 'rbf', probability=True)     # classifier with best parameters
    model_fit = model.fit(X_train,y_train)                                     # fit data to model
    predicted=model_fit.predict(X_test)                                        # predict classes for test data

    print("\n__________________________________________\n")
    f1 = f1_score(y_test, predicted, average='micro')                          # f1 accuracy score
    cm = confusion_matrix(y_test, predicted)                                   # generate confusion matrix for validation set 
    print("Validation Set F1 score:",f1)                                       # print scores
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot()
    
    # save model and scaler
    # pickle.dump(model_fit, open('models/svm_model.sav', 'wb'))
    # pickle.dump(sc, open('models/svm_scaler.pkl', 'wb'))

    
    
def runModel(X_train, y_train, X_test, y_test,le):
    """
    Function to run process of creating and testing an SVM classifier on this dataset using loaded extracted features
    """
    #Feature Scaling
    print("Scaling")
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #Model Building

    # search for best parameters    
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
    
    model = svm.SVC(C = 1500, gamma = 1, kernel = 'rbf', probability=True)     # classifier with best parameters
    model_fit = model.fit(X_train,y_train)                                     # fit data to model
    predicted=model_fit.predict(X_test)                                        # predict classes for test data

    print("\n__________________________________________\n")
    f1 = f1_score(y_test, predicted, average='micro')                          # f1 accuracy score
    cm = confusion_matrix(y_test, predicted)                                   # generate confusion matrix for validation set 
    print("Validation Set F1 score:",f1)                                       # print scores
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot()
    
    # save model and scaler
    # pickle.dump(model_fit, open('models/svm_model.sav', 'wb'))
    # pickle.dump(sc, open('models/svm_scaler.pkl', 'wb'))
    

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = loadData()  # load pre-extracted features
    
    le = preprocessing.LabelEncoder()
    le.fit(["Door Knocking","Shower Running","Toilet Flushing","Vacuum Cleaning","Keyboard Typing",  # encode class labels as numeric id values
                "Coughing","Neutral"])
    runModel(X_train, y_train, X_test, y_test, le) # fit svm classifier 
    #runAll()                                      # extract features and fit svm classifier
    #df = test()                                   # test classifier on test wavs
    
    
    
    
    
# # save the model to disk
# pickle.dump(best_model, open('models/svm_model.sav', 'wb'))
# pickle.dump(sc, open('models/svm_scaler.pkl', 'wb'))
 
 
# # load the model from disk
# loaded_model = pickle.load(open("SVM_model.sav", 'rb'))
# result = loaded_model.score(X_test, y_test)
# print(result)