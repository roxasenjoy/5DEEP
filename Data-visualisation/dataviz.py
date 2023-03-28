import os

import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt


def get_audio(amountAudioPerFile=0, getGraphics=False):
    link_folder = "../UrbanSound8K/audio/"
    extension = ".wav"
    number_of_folder = 11
    index_file = 0

    for folderIndex in range(1, number_of_folder):
        folder_name = "fold" + str(folderIndex) + "/"
        path = link_folder + folder_name
        for file in os.listdir(path):  # On passe sur tous les audios du dossier foldX
            if file.endswith(extension):  # Vérification de l'extension du
                index_file += 1
                audioPath = path + file

                # 2. Load the audio as a waveform `y`
                # Store the sampling rate as `sr`
                audio, sampling_rate = librosa.load(audioPath)

                time = np.arange(0, len(audio)) / sampling_rate
                audio = pd.DataFrame(audio)

                if getGraphics:
                    # plt.set_ylim([0, 1])
                    plt.ylim(-1, 1)
                    plt.subplot(5, 2, folderIndex)
                    plt.plot(time, audio)  # X / Y
                    plt.title("" + str(folderIndex))
                    plt.xlabel('Temps en s')
                    plt.ylabel('Amplitude')

                # Condition pour éviter de sélectionner tous les éléments des dossiers car trop long
                if index_file == amountAudioPerFile:
                    index_file = 0
                    break

    if getGraphics:
        plt.show()

get_audio(1, True)
