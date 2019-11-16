from itertools import zip_longest
import librosa as lr
import numpy as np
import glob
import os
import math
import tqdm

audio_dir = os.path.join("/Users", "preetham", "PycharmProjects", "Audio_Classification", "RW_AUDIO_DATA_2019_Update")
train_dir = os.path.join("/Users", "preetham", "PycharmProjects", "Audio_Classification", "RW_AUDIO_DATA_2019_Update",
                         "JUNE_01_PHANTOMS")
negative_train_dir = os.path.join("/Users", "preetham", "PycharmProjects", "Audio_Classification",
                                  "RW_AUDIO_DATA_2019_Update", "JUNE_02_BACKGROUND")

audio_files = glob.glob(
    "/Users/preetham/PycharmProjects/Audio_Classification/RW_AUDIO_DATA_2019_Update/JUNE_01_PHANTOMS/*.wav")
bg_files = glob.glob(
    "/Users/preetham/PycharmProjects/Audio_Classification/RW_AUDIO_DATA_2019_Update/JUNE_02_BACKGROUND/*.wav")


def load_file(file_name, sampling_rate=16000):
    return lr.core.load(file_name, sr=sampling_rate)


def split_files(src_file, dest_loc, n_splits):
    data, sr = load_file(src_file)
    n_splits = math.floor(len(data) / (16000 * 3))
    if not os.path.exists(dest_loc):
        os.makedirs(dest_loc)
    dest_loc = os.path.join(dest_loc, os.path.basename(src_file))
    for i, split in enumerate(np.array_split(data, n_splits), start=1):
        lr.output.write_wav(dest_loc + "_split_" + str(i), y=split, sr=16000)
        print(dest_loc + "_split_" + str(i) + "  created!")


def generate_split_files(audio_files,bg_files):
    for file1, file2 in zip_longest(audio_files, bg_files):
        if file1 is not None:
            split_files(file1, "processed_audio")
        if file2 is not None:
            split_files(file2, "processed_audio")

def squarize(vector):
    """
    converts non-square matrix to a square matrix
    """
    m,n = vector.shape
    if m > n:
        pad = np.zeros(shape=(m, m-n))
        return np.c_[vector,pad]
    elif n > m:
        pad = np.zeros(shape=(n-m, n))
        return np.r_[vector,pad]
    return vector



if __name__ == "__main__":
    pass
