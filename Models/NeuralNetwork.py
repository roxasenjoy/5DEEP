import keras
import tensorflow as tf
import keras.optimizers
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import numpy as np
import os
from keras.models import load_model

from sklearn.model_selection import train_test_split


class NeuralNetwork(ABC):
    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def predict(self, features):
        pass

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def getInfos(self):
        pass

    @abstractmethod
    def graphForTraining(self):
        pass



class ConvolutionalNetwork(NeuralNetwork):
    def __init__(self, model, name, X, Y, option):
        self.name = name
        self.model = model
        self.history = None
        self.isTrained = False
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.score = []
        self.preprocessorOption = option


        listFile = os.listdir('Models/')
        for index, file in enumerate(listFile):
            if file == (name + ".hdf5"):
                print("Loading model ...")
                self.model = load_model('Models/' + name + ".hdf5")
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                self.score = self.model.evaluate(self.X_test, self.y_test, verbose=0)
                self.isTrained = True


    def fit(self, X, Y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto',
            baseline=None, restore_best_weights=True)

        self.history = self.model.fit(self.X_train, self.y_train,
                            validation_data=(self.X_test, self.y_test),
                            epochs=15,
                            callbacks=[callback],
                            batch_size=64)
        self.isTrained = True

        self.score = self.model.evaluate(self.X_test, self.y_test, verbose=1)

        self.model.save("Models/" + self.name + ".hdf5", True, True)

    def compile(self):
        opt = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics='acc')

    def predict(self, features):
        pred = self.model.predict(features)
        prediction = np.argmax(pred, axis=-1)
        return prediction

    def graphForTraining(self):
        if self.history != None:
            plt.plot(self.history.history['acc'])
            plt.plot(self.history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.ylim([0.0, 1.0])
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()

            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')

            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
        else :
            print("\nIl faut entrainer votre modèle pour pouvoir voir ces données\n")


    def summary(self):
        self.model.summary()

    def getInfos(self):
        return self.name, self.isTrained, self.score


class FullyConnectedNetwork(NeuralNetwork):
    def __init__(self, model, name, X, Y, option):
        self.name = name
        self.model = model
        self.history = None
        self.isTrained = False
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.score = []
        self.preprocessorOption = option


        listFile = os.listdir('Models/')
        for index, file in enumerate(listFile):
            if file == (name + ".hdf5"):
                print("Loading model ...")
                self.model = load_model('Models/' + name + ".hdf5")
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                self.score = self.model.evaluate(self.X_test, self.y_test, verbose=0)
                self.isTrained = True


    def fit(self, X, Y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        self.history = self.model.fit(self.X_train, self.y_train,
                            validation_data=(self.X_test, self.y_test),
                            epochs=50,
                            batch_size=10)
        self.isTrained = True

        self.score = self.model.evaluate(self.X_test, self.y_test, verbose=1)

        self.model.save("Models/" + self.name + ".hdf5", True, True)

    def compile(self):
        opt = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics='acc')

    def predict(self, features):
        pred = self.model.predict(features)
        prediction = np.argmax(pred, axis=-1)
        return prediction

    def graphForTraining(self):
        if self.history != None:
            plt.plot(self.history.history['acc'])
            plt.plot(self.history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.ylim([0.0, 1.0])
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()

            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')

            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
        else :
            print("\nIl faut entrainer votre modèle pour pouvoir voir ces données\n")


    def summary(self):
        self.model.summary()

    def getInfos(self):
        return self.name, self.isTrained, self.score



