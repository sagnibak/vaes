import array
import sys; sys.path.append('/home/sagnick/PycharmProjects/vaes')
from utils.sounds import dct_by_interval, idct_by_interval, AudioSegment

song = AudioSegment.from_mp3('/home/sagnick/Downloads/audio_data/moiaimertoi.mp3')
dcted = dct_by_interval(song, 8000)
repro = idct_by_interval(dcted)
repro_arr = array.array(song.array_type, repro.reshape(-1))
print('Shape of reproduced sound is ', repro.shape)
repro_song = song._spawn(repro_arr)
repro_song.export('../out_sounds/moiaimertoi_interval_dct.wav', format='wav')
