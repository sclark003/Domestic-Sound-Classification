import numpy as np
import pandas as pd



def predictFramewise(predicted, audio_files, le):
    estimated_event_list = []
    predicted = predicted.detach().cpu().numpy()
    class_id = np.bincount(predicted).argmax()
    
    for i in range(len(predicted)):
        prediction = predicted[i]
        audio_id = audio_files[i]
        audio_id = audio_id.split("_")
        audio_id = audio_id[0]
        if audio_id[-4:]=='.wav':
            audio_id = audio_id[0:-4]
        #num_id = audio_id.split("_")
        #num_id = num_id[1]
        num_id = i+1
        if prediction != class_id:
            pass
        else:
            estimated_event = {
                "audio_id": audio_id,
                "class": le.inverse_transform([prediction]),
                "onset": int(num_id),
                "offset": int(num_id) + 1,
                }
            estimated_event_list.append(estimated_event)
    return estimated_event_list, le.inverse_transform([class_id])
        

def combineEvents(e):
    for i in range(len(e)):
        length = len(e)
        if i < length-1:
            event = e[i]
            next_event = e[i+1]
            if event["offset"] == next_event["onset"]:
                next_event["onset"] = event["onset"]
                del e[i]
    return e

def checkEvents(e):
    for i in range(len(e)-1):
        event = e[i]
        next_event = e[i+1]
        if (event["offset"] == next_event["onset"]):
            return True
    return False

def postProcess(e):
    combine = True
    while combine:
        e = combineEvents(e)
        combine = checkEvents(e)
    return e