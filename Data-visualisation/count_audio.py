import os

def count_audio_in_folder():
    link_folder = "../UrbanSound8K/audio/"
    extension = ".wav"
    number_of_folder = 11
    total = 0
    count_audio_array = []

    for folderIndex in range(1, number_of_folder):
        folder_name = "fold" + str(folderIndex) + "/"
        path = link_folder + folder_name
        for file in os.listdir(path):
            if file.endswith(extension):
                total += 1

        count_audio_array.append({'Fold' + str(folderIndex): total})
        total = 0
    print(count_audio_array)


print('#6 - Est ce que le dataset est équilibré ?')
count_audio_in_folder()