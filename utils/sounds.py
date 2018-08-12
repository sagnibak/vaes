import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from scipy.fftpack import dct, idct


def dct_by_interval(sound: AudioSegment, samples_per_interval: int):
    """Calculates the Discrete Cosine Transform of `sound` by
    taking `samples_per_interval` samples in each interval.

    :param sound:                 AudioSegment to calculate the DCT of
    :param samples_per_interval:  number of samples to include in each
                                  DCT interval
    :return:  Numpy ndarray with shape
                (num_intervals, samples_per_interval, 2)
    """
    sound_arr = np.array(sound.get_array_of_samples()).reshape(-1, 2).T

    pad_length = samples_per_interval - (sound_arr.shape[1] % samples_per_interval)
    sound_arr = np.pad(sound_arr, ((0, 0), (0, pad_length)),
                       'constant', constant_values=0)

    num_intervals = sound_arr.shape[1] // samples_per_interval
    result = np.empty((num_intervals, samples_per_interval, 2), dtype=np.float32)

    for i in range(num_intervals):
        result[i, :, :] = dct(sound_arr[:, i * samples_per_interval:
                                           (i + 1) * samples_per_interval],
                              norm='ortho').T
    return result


def idct_by_interval(dcted: np.array):
    """Calculates the Inverse Dicrete Cosine Transform of `DCTed` by
    taking `samples_per_interval` samples in each interval.

    :param dcted:  Output of `dct_by_interval` or equivalent Numpy ndarray
                   with shape (num_intervals, samples_per_interval, 2)
    :return:  Numpy ndarry of shape (num_intervals * samples_per_interval, 2)
              that can be converted to stereo audio using pydub after doing
              output.reshape(-1)
    """
    result = np.empty((2, dcted.shape[0] * dcted.shape[1]), dtype=np.int16)

    for i in range(dcted.shape[0]):
        temp = idct(dcted[i].T, norm='ortho')
        temp = temp.astype(np.int16)  # Contrary to what I had suspected, x ~= idct(dct(x)),
                                      # so there isn't much concern about rollover
        result[:, i * dcted.shape[1]: (i + 1) * dcted.shape[1]] = temp.astype(np.int16)

    return result.T  # so that you can do result.reshape(-1) while converting it into an
                     # `array.array` instance


def plot_sound_dct(dct_result: np.array, save_name: str=None):
    """Plot the result of the function `dct_by_interval`, which is the
    Discrete Cosine Transform of a song (pydub.AudioSegment instance)
    taken over intervals of time.

    :param dct_result: output of function `dct_by_interval` or equivalent Numpy
                       ndarray of shape (num_intervals, samples_per_interval, 2)
    :param save_name:  filename to save the figure
    """

    def thresh(x, threshold):
        y = np.copy(x)
        y[abs(y) < threshold] = 0
        return y

    left = thresh(dct_result[:, :, 0].T, 2000)
    right = thresh(dct_result[:, :, 1].T, 2000)

    plt.figure(figsize=(20, 8))
    plt.matshow(left, aspect='auto', cmap='nipy_spectral')
    plt.colorbar()

    if save_name is not None:
        save_name_left = save_name + f'_{dct_result.shape[1]}_intervals_left.png'
        plt.savefig(f'visualization/{save_name_left}')

    plt.show()

    plt.figure(figsize=(20, 8))
    plt.matshow(right, aspect='auto', cmap='nipy_spectral')
    plt.colorbar()

    if save_name is not None:
        save_name_right = save_name + f'_{dct_result.shape[1]}_intervals_right.png'
        plt.savefig(f'visualization/{save_name_right}')

    plt.show()


if __name__ == '__main__':
    print('Made it here in one piece!!!')
