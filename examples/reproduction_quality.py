"""In conclusion, keeping only the first eighth of the DCT
elements is good enough. It sounds a little muffled, and there
are crackling artifacts in the reproduced song, but the songs
sound good enough--this seems like the best compromise."""

import array
import os
from pydub import AudioSegment
import sys; sys.path.append('/home/sagnick/PycharmProjects/vaes')
from utils.sounds import dct_by_interval, idct_by_interval

SONG_DIR = '/home/sagnick/Downloads/audio_data/'

song_name = input('Which song do you want to ~compress~? ')
samples_per_interval = eval(input(f'How many samples do you want in each DCT interval for "{song_name}"? '))

song = AudioSegment.from_mp3(f'{os.path.join(SONG_DIR, song_name)}.mp3')
dct_result = dct_by_interval(song, samples_per_interval)
dct_result[:, (samples_per_interval//8):, :] = 0

repro = idct_by_interval(dct_result)
repro_arr = array.array(song.array_type, repro.reshape(-1))
repro_song = song._spawn(repro_arr)
save_path = f'out_sounds/{song_name}_{samples_per_interval}_i_1_8.wav'
repro_song.export(save_path, format='wav')

print(f'Saved reproduced song to {save_path}')
