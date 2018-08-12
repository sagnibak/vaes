import os
from pydub import AudioSegment
import sys; sys.path.append('/home/sagnick/PycharmProjects/vaes')
from utils.sounds import dct_by_interval, plot_sound_dct

SONG_DIR = '/home/sagnick/Downloads/audio_data/'
song_name = input('Which song do you want to view the DCT of? ')

samples_per_interval = eval(input(f'How many samples do you want in each DCT interval for "{song_name}"? '))

song = AudioSegment.from_mp3(f'{os.path.join(SONG_DIR, song_name)}.mp3')
dct_result = dct_by_interval(song, samples_per_interval)

plot_sound_dct(dct_result, f'{song_name}_vis')
