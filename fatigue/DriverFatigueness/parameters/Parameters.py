'''
All parameters handling happens here
'''
from __future__ import print_function
import os
from utils.properties import Properties
from utils.propertiesUtils import getPathAtFolder, converter, getPropertiesAtFolder


class TestEnvParameters:

    def __init__(self,propFolder=""):
        try:
            path = getPropertiesAtFolder(propFolder,"test.properties")

        except:
            print("Path not found in local folder. Going to resources!")
            path = getPathAtFolder("resources", "test.properties")

        prop = Properties()
        prop.load(open(path))


        # path variables
        self.DIR_PATH=str(os.getcwd())
        print('Working in :',self.DIR_PATH)
        self.HAAR_PATH = self.tryCatch(prop,'haar_dir', 's', '')
        self.VIDEO_PATH = self.tryCatch(prop,'vid_path', 's', 'device(0)')

        # CNN Model training
        self.EPOCHS_EYE = self.tryCatch(prop, 'epochs_eye','i',5)
        self.EPOCHS_YAWN = self.tryCatch(prop, 'epochs_yawn','i',5)

        # Use Pretrained or Saved models
        self.USE_SAVED_MODEL_EYE = self.tryCatch(prop, 'saved_model_eye', 'b', True)
        self.USE_SAVED_MODEL_PERCLOS = self.tryCatch(prop, 'saved_model_perclos', 'b', True)
        self.USE_SAVED_MODEL_YAWN = self.tryCatch(prop, 'saved_model_yawn', 'b', True)
        self.USE_SAVED_MODEL_DROWSINESS = self.tryCatch(prop, 'saved_model_drowsiness', 'b', True)


        # Probability Cutoffs for Blink and Yawn
        self.PROB_EYE = self.tryCatch(prop,'prob_eye', 'f', 0.6)
        self.PROB_YAWN = self.tryCatch(prop,'prob_yawn', 'f', 0.05)
        self.PROB_PERCLOS = self.tryCatch(prop, 'prob_perclos', 'f', 0.5)
        self.PROB_DROWSY = self.tryCatch(prop, 'prob_drowsiness', 'f', 0.5)

        # Wait time for Evaluation
        self.WAIT_TIME = self.tryCatch(prop,'wait_time', 'i', 200)
        self.INTER_WAIT_TIME = self.tryCatch(prop,'inter_wait_time', 'i', 1000)

        # Drowsiness criterion

        ## Frames to detect
        self.SECONDS_TO_BLINK = self.tryCatch(prop,'seconds_to_blink', 'f', 0.1)
        self.SECONDS_TO_PERCLOS = self.tryCatch(prop, 'seconds_to_blink', 'f', 0.1)
        self.SECONDS_TO_YAWN = self.tryCatch(prop,'seconds_to_yawn', 'f', 1.2)

        ## Drowsiness Counters
        self.BLINK_COUNTER_WINDOW = self.tryCatch(prop,'blink_counter_window', 'i',30)
        self.PERCLOS_COUNTER_WINDOW = self.tryCatch(prop,'perclos_counter_window', 'i', 180)
        self.YAWN_COUNTER_WINDOW = self.tryCatch(prop,'yawn_counter_window', 'i', 120)
        self.COUNTER_INITIAL_VALUE = self.tryCatch(prop,'counter_initial_value', 'i', 0)

        # UI parameters
        self.VIDEO_RESOLUTION = self.tryCatch(prop,'resolution_of_video_screen','i',400)

        # Saving options
        self.SAVE_OUTPUT = self.tryCatch(prop, 'save_output', 'b', True)
        self.OUTPUT_FILE = self.tryCatch(prop, 'output_file', 's', 'data_{}.csv'.format('default'))

        # Logging options
        try:
            self.ALLOW_LOGGING=converter(prop.getProperty('allow_logging'),"b")
        except:
            self.ALLOW_LOGGING=False

        if self.ALLOW_LOGGING:
            try:
                self.LOGGER_LEVEL=converter(prop.getProperty('logging_level'))
            except:
                self.LOGGER_LEVEL="INFO"

    def __repr__(self):
        return [self.VIDEO_PATH, self.HAAR_PATH, self.EPOCHS_YAWN, self.EPOCHS_EYE, self.INTER_WAIT_TIME,
                self.WAIT_TIME, self.PROB_EYE, self.PROB_YAWN, self.SECONDS_TO_BLINK, self.ALLOW_LOGGING]

    def tryCatch(self,prop, value, selector, default):
        try:
            val = converter(prop.getProperty(value),selector)
            if val ==  None:
                raise Exception('Value not found returning default value')
            print('Got Value for', value, 'as', val)
            return val
        except:
            print('Returning default for', value, 'as', default)
            return default


if __name__ == '__main__':
    dirPath=os.getcwd()
    dirPath
    param = TestEnvParameters(propFolder=os.getcwd())
    print('Properties have values:', param.__repr__())
