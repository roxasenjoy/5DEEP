import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter


def get_canal_audio(amountAudioPerFile=0, getGraphics=False):
    link_folder = "../UrbanSound8K/audio/"
    extension = ".wav"
    number_of_folder = 11
    index_file = 0
    canalData = []
    frequencyData = []

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
                channel = audio.shape[0]

                # Calculer la distribution de fréquence d'échantillonnage
                stft = librosa.stft(audio)
                spectrum, phase = librosa.magphase(stft)

                if getGraphics:

                    librosa.display.specshow(librosa.amplitude_to_db(spectrum, ref=np.max), sr=sampling_rate,y_axis='log', x_axis='time')
                    plt.colorbar()
                    plt.title('Distribution de la fréquence d\'échantillonnage - Folder ' + str(folderIndex))
                    plt.tight_layout()
                    plt.show()

                # Créer une liste avec tous les éléments du fichier
                # Compter le nombre d'occurence
                canalData.append(channel)
                frequencyData.append(spectrum)

                # Condition pour éviter de sélectionner tous les éléments des dossiers car trop long
                if index_file == amountAudioPerFile:
                    index_file = 0
                    break



    return {
        'canalData': canalData,
        'frequencyData': frequencyData
    }


# 7 - Distribution du nombre de canaux de chaque son (les 1er de chaque son)
plt.hist(get_canal_audio(1, False)['canalData'], color='blue')
plt.xlabel('Nombre canal')
plt.ylabel('Total')
plt.title('Distribution du nombre de canaux')
plt.show()

# 8 - Fréquence d'échantillonage de chaque son (les 1er de chaque type)
frequencyData = get_canal_audio(1, True)['frequencyData']
print(type(frequencyData))
df = pd.DataFrame(frequencyData[0])
print(df)


