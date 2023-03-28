import librosa
import pandas as pd
import progressbar
import os
import constant
from constant import PreprocessorOption
import cv2
import numpy as np
from numpy import asarray, load
from numpy import save
import speechpy
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

class Preprocessor:
    def __init__(self, path):
        self.path = path
        self.fileName = None
        self.data = None
        self.bar = None
        self.X1d = []
        self.X2d = []
        self.Y = []
        self.labelEncoder = LabelEncoder()
        pass

    def openFile(self, fileName):
        self.fileName = fileName
        self.data = pd.read_csv(self.path + self.fileName)

    def parseFile(self):
        if os.path.exists("data/dataX1D.npy") and os.path.exists("data/dataY.npy") and os.path.exists("data/dataX2D.npy"):
            self.X1d = load('data/dataX1D.npy')
            self.X2d = load('data/dataX2D.npy')
            self.Y = load('data/dataY.npy')

        elif not self.data.empty:
            X = []
            self.X1d = []
            self.X2d = []
            barProgression = 0
            self.loadProgressBar()
            self.bar.start()
            for data in self.data.iterrows():
                file_name = os.path.join(os.path.abspath(constant.AUDIO_PATH), 'fold' + str(data[1][5]) + '/', str(data[1][0]))
                try:
                    raw, sr = librosa.load(file_name, res_type='kaiser_fast')
                    X = librosa.feature.mfcc(y=raw, sr=sr, n_mfcc=constant.HEIGHT)
                    up_points = (constant.WIDTH, constant.HEIGHT)
                    X = cv2.resize(X, up_points, interpolation=cv2.INTER_LINEAR)
                    X = speechpy.processing.cmvn(X)
                    X_ = np.mean(X.T, axis=0)
                    self.X2d.append(X)
                    self.X1d.append(X_)
                    self.Y.append(data[1][7])

                except Exception as e:
                    print(e)

                barProgression += 1
                self.bar.update(barProgression)

            self.bar.finish()
            self.X1d = np.array(self.X1d)
            self.X2d = np.array(self.X2d)
            self.y = np.array(self.Y)

            self.Y = to_categorical(self.labelEncoder.fit_transform(self.Y))
            self.X2d = self.X2d.reshape(self.X2d.shape[0], self.X2d.shape[1], self.X2d.shape[2])
            print(self.X2d.shape)
            save("data/dataX1D.npy", self.X1d)
            save("data/dataX2D.npy", self.X2d)
            save("data/dataY.npy", self.Y)

        return self.X1d, self.X2d, self.Y


    def loadProgressBar(self):
        widgets = [' [',
           progressbar.Timer(format='elapsed time: %(elapsed)s'),
           '] ',
           progressbar.Bar('■'), ' (',
           progressbar.ETA(), ') ',
        ]

        self.bar = progressbar.ProgressBar(max_value=len(self.data.index), widgets=widgets)


    def getFeaturesForTestSound(self, fileName, option):
        raw, sr = librosa.load(constant.TEST_PATH + fileName, res_type='kaiser_fast')
        X_sound = librosa.feature.mfcc(y=raw, sr=sr, n_mfcc=constant.HEIGHT)
        up_points = (constant.WIDTH, constant.HEIGHT)
        X_sound = cv2.resize(X_sound, up_points, interpolation=cv2.INTER_LINEAR)
        X_sound = speechpy.processing.cmvn(X_sound)
        if option == PreprocessorOption.singleDimension:
            X_sound = np.mean(X_sound.T, axis=0)
            X_sound = X_sound.reshape(1, X_sound.shape[0])
        elif option == PreprocessorOption.doubleDimension :
            X_sound = X_sound.reshape(1, X_sound.shape[0], X_sound.shape[1])

        return X_sound