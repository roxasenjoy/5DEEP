import os
import pandas as pd


def count_audio_in_folder():
    df = pd.read_csv('../UrbanSound8K/metadata/UrbanSound8K.csv')
    count_audio = df.groupby(by=['class'])['class'].count()
    print(count_audio)


print('#6 - Est ce que le dataset est équilibré ?')
count_audio_in_folder()
