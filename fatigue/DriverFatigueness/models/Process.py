'''
This file is for Loading/Training the CNN models and Testing/Predicting CNN on the Input Images. 
'''

import numpy as np
from models.YawnCNN import yawnCNNModel
from models.EyesCNN import eyeCNNModel
from models.PerclosCNN import perclosCNNModel
from keras.models import load_model, model_from_json
from utils.Utils import resizing

eyeModel = None
yawnModel = None
perclosModel = None
drowsinessModel = None


def modelLoader(epochs_eye=12, epochs_yawn=20, saved_models=[True, True, True, True], model_path=''):

    # Loading / Generating Eye CNN Model
    global eyeModel
    if saved_models[0]:
        try:
            print('Loading Blink CNN Model!')
            eyeModel = loadModel('{}/{}_std'.format(model_path, "eyeCNN"))
            print("Eye Model Loaded as", eyeModel.summary())
        except:
            print("No Eye Model found! Regenerating standard model")
            eyeModel = eyeCNNModel()
            print("Saving model at: ", '{}/{}_std'.format(model_path, "eyeCNN"))
            saveModel(eyeModel, '{}/{}_std'.format(model_path, "eyeCNN"))
            print("Eye Model Loaded as ", eyeModel.summary())
    else:
        eyeModel = eyeCNNModel(epochs=epochs_eye)
        print("Saving model at: {}", '{}/{}_{}'.format(model_path, "eyeCNN",epochs_eye))
        saveModel(eyeModel, '{}/{}_{}'.format(model_path, "eyeCNN",epochs_eye))
        # eyeModel.save('{}/{}'.format(model_path,"eyeCNN.h5"))
        print("Eye Model Loaded as", eyeModel.summary())

    # Loading / Generating Perclos CNN Model
    global perclosModel
    if saved_models[0]:
        try:
            print('Loading PERCLOS CNN Model!')
            perclosModel = loadModel('{}/{}_std'.format(model_path, "perclosCNN"))
            print("Eye Model Loaded as", perclosModel.summary())
        except:
            print("No Perclos Model found! Regenerating standard model")
            perclosModel = perclosCNNModel()
            print("Saving model at: ", '{}/{}_std'.format(model_path, "perclosCNN"))
            saveModel(perclosModel, '{}/{}_std'.format(model_path, "perclosCNN"))
            print("Perclos Model Loaded as", perclosModel.summary())
    else:
        perclosModel = eyeCNNModel(epochs=epochs_eye)
        print("Saving model at: ", '{}/{}'.format(model_path, "perclosCNN"))
        saveModel(perclosModel, '{}/{}'.format(model_path, "perclosCNN"))
        # eyeModel.save('{}/{}'.format(model_path,"eyeCNN.h5"))
        print("Perclos Model Loaded as", perclosModel.summary())

    # Loading / Generating Yawn CNN Model
    global yawnModel
    if saved_models[2]:
        try:
            print('Loading Yawn CNN Model!')
            yawnModel = loadModel('{}/{}_std'.format(model_path, "yawnCNN"))
            print("Yawn Model Loaded as", yawnModel.summary())
        except:
            print('No Yawn Model found! Regenerating standard model')
            yawnModel = yawnCNNModel()
            print("Saving model at: ", '{}/{}_std'.format(model_path, 'yawnCNN'))
            saveModel(yawnModel, '{}/{}_std'.format(model_path, "yawnCNN"))
            print("yawn Model Loaded as", yawnModel.summary())
    else:
        yawnModel = eyeCNNModel(epochs=epochs_eye)
        print('Saving model at: ', '{}/{}_{}'.format(model_path, 'yawnCNN', epochs_yawn))
        saveModel(yawnModel, '{}/{}_{}'.format(model_path, "yawnCNN",epochs_yawn))
        # eyeModel.save('{}/{}'.format(model_path,"eyeCNN.h5"))
        print("Yawn Model Loaded as", yawnModel.summary())




def testEyeOnModel(input_image):
    X_pred = np.ndarray([1, 24, 24, 1], dtype='float32')
    im= np.dot(np.array(input_image, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255
    X_pred[0, :, :, 0] = resizing(im, 24, maintain_aspect = False)
    prob_eye = eyeModel.predict(X_pred)
    return(prob_eye)

def testPerclosOnModel(input_image):
    X_pred = np.ndarray([1, 24, 24, 1], dtype='float32')
    im= np.dot(np.array(input_image, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255
    X_pred[0, :, :, 0] = resizing(im, 24, maintain_aspect = False)
    prob_perclos = perclosModel.predict(X_pred)
    return(prob_perclos)

def testYawnOnModel(input_image):
    X_pred = np.ndarray([1, 60, 60, 1], dtype='float32')
    im= np.dot(np.array(input_image, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255
    X_pred[0, :, :, 0] = resizing(im, 60, maintain_aspect = False)
    prob_yawn = yawnModel.predict(X_pred)
    return(prob_yawn)

def testDrowsieness(eyeCounter,perclosCounter,yawnCounter):
    X_pred = np.ndarray([1, 24, 24, 1], dtype='float32')
    im= np.dot(np.array(input_image, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255
    X_pred[0, :, :, 0] = resizing(im, 24, maintain_aspect = False)
    prob_eye = eyeModel.predict(X_pred)
    return(prob_eye)

def saveModel(model,name):
    json_model = model.to_json()
    with open('{}.json'.format(name), "w") as json_file:
        json_file.write(json_model)

    model.save_weights('{}.h5'.format(name))

def loadModel(name):
    json_file = open('{}.json'.format(name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('{}.h5'.format(name))
    return model

if __name__ == '__main__':

    # eyeModel = eyeCNNModel()
    # yawnModel = yawnCNNModel()

    modelLoader()
