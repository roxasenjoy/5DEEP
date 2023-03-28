import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Objectif : Récupérer la distribution de chaque son
def distribution_all_sounds():
    df = pd.read_csv('../UrbanSound8K/metadata/UrbanSound8K.csv')
    listChannel = []

    for index, row in df.iterrows():
        audio, sampling_rate = librosa.load(
            '../UrbanSound8K/audio/fold' + str(row['fold']) + '/' + str(row['slice_file_name']))
        listChannel.append(audio.shape[0])
        print(index)

    return listChannel


x = distribution_all_sounds()
newDf = pd.DataFrame(x, columns= ['Nombre de canaux'])
# 7 - Distribution du nombre de canaux de chaque son
plt.hist(x, color='blue')
plt.xlabel('Nombre de canaux')
plt.ylabel('Total')
plt.title('Distribution du nombre de canaux')
plt.show()

# ne fonctionne pas sans raison :
df = pd.DataFrame(x, columns=['Nombre de canaux'])
df.hist(column='Nombre de canaux')

# 8 - Fréquence d'échantillonage de chaque son (les 1er de chaque type)
# frequencyData = get_canal_audio(2, True)['frequencyData']
# print(type(frequencyData))
# print(frequencyData)
# df = pd.DataFrame(frequencyData[2])
# print(df)
