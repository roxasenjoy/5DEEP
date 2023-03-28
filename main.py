

import os
import constant
from constant import PreprocessorOption
from Models.NeuralNetwork import ConvolutionalNetwork
from Models.NeuralNetwork import FullyConnectedNetwork
from Preprocessor import Preprocessor

class Program:
    def __init__(self):
        self.stopProgram = False
        self.preprocessor = Preprocessor("UrbanSound8K/metadata/")

        self.preprocessor.openFile("UrbanSound8K.csv")
        self.X1d, self.X2d, self.Y = self.preprocessor.parseFile()

        self.models = [FullyConnectedNetwork(constant.FULLYCONNECTED_MODEL, "fullyConnectedModel1", self.X1d, self.Y, PreprocessorOption.singleDimension),
                        ConvolutionalNetwork(constant.CONVOLUTIONAL_MODEL1, "convolutionalModel1", self.X2d, self.Y, PreprocessorOption.doubleDimension),
                        ConvolutionalNetwork(constant.CONVOLUTIONAL_MODEL2, "convolutionalModel2", self.X2d, self.Y, PreprocessorOption.doubleDimension),
                        ConvolutionalNetwork(constant.CONVOLUTIONAL_MODEL3, "convolutionalModel3", self.X2d, self.Y, PreprocessorOption.doubleDimension)]

    def run(self):
        while not self.stopProgram:
            print("\n\nQue voulez vous faire \n"
                  "1) Résultat de l'analyse des données\n"
                  "2) Re.lancer le prétraitement des données\n"
                  "3) Afficher les modèle disponible\n"
                  "4) Entrainer un modèle\n"
                  "5) Prédire la catégorie d'une nouvelle musique\n"
                  "6) Afficher les graphiques lié à l'entrainement d'un modèle\n"
                  "7) Afficher les détails d'un modèle\n"
                  "8) arrêter le programme\n")

            response = input("réponse : ")

            if response == "1":
                # TODO Andrew
                pass

            elif response == "2":
                self.preprocessor.openFile("UrbanSound8K.csv")
                self.X1d, self.X2d, self.Y = self.preprocessor.parseFile()
                print("finished")

            elif response == "3":
                for index, model in enumerate(self.models):
                    name, istrained, score = model.getInfos()
                    print(str(index+1) + ") " + name + (" (acc="+ str(score[1]) + " loss="+ str(score[0]) +") " if istrained else " (not trained)"))

            elif response == "4":
                if not self.X1d.size == 0 and not self.X2d.size == 0 and not self.Y.size == 0:
                    print("Quel modèle voulez-vous entrainer ?")
                    for index, model in enumerate(self.models):
                        name, istrained, score = model.getInfos()
                        print(str(index + 1) + ") " + name + (" (acc="+ str(score[1]) + " loss="+ str(score[0]) +") " if istrained else " (not trained)"))

                    choice = input("\nchoix : ")
                    self.models[int(choice) - 1].compile()
                    if self.models[int(choice) - 1].preprocessorOption == PreprocessorOption.singleDimension:
                        self.models[int(choice) - 1].fit(self.X1d, self.Y)
                    else :
                        self.models[int(choice) - 1].fit(self.X2d, self.Y)

                else:
                    print("Vous devez d'abord faire le prétraitement des données")

            elif response == "5":
                print("\nAvec quel model ?\n")
                for index, model in enumerate(self.models):
                    name, istrained, score = model.getInfos()
                    print(str(index + 1) + ") " + name + (
                        " (acc=" + str(score[1]) + " loss=" + str(score[0]) + ") " if istrained else " (not trained)"))

                choice2 = input("\nchoix : \n")


                listFile = os.listdir('test/')
                print("Quel son voulez vous tester ?")
                for index, file in enumerate(listFile):
                    print(str(index+1) + ") " + file)

                choice = input("\nchoix : ")
                features = self.preprocessor.getFeaturesForTestSound(listFile[int(choice)-1], self.models[int(choice2) - 1].preprocessorOption)

                pred = self.models[int(choice2) - 1].predict(features)
                print(constant.CATEGORIES[pred[0]])



            elif response == "6":
                print("\nde quel modèle voulez-vous afficher les graphiques ?\n")
                for index, model in enumerate(self.models):
                    name, istrained, score = model.getInfos()
                    print(str(index + 1) + ") " + name + (" (acc=" + str(score[1]) + " loss=" + str(score[0]) + ") " if istrained else " (not trained)"))

                choice = input("\nchoix : ")
                self.models[int(choice) - 1].graphForTraining()

            elif response == "7":
                print("\nde quel modèle voulez-vous afficher les détails ?\n")
                for index, model in enumerate(self.models):
                    name, istrained, score = model.getInfos()
                    print(str(index + 1) + ") " + name + (" (acc=" + str(score[1]) + " loss=" + str(score[0]) + ") " if istrained else " (not trained)"))

                choice = input("\nchoix : ")
                self.models[int(choice) - 1].summary()

            elif response == "8":
                self.stopProgram = True


program = Program()
program.run()