from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import mean_squared_error
from keras.models import Model
from keras.regularizers import l2

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from utils.sounds import dct_by_interval, idct_by_interval


def sampling(args):
    z_mean, z_log_var = args
    raise NotImplementedError('Learn how to sample (maybe from Costco lol)')


def conv_bn_lrelu(x, filters, kernel_size, strides):
    """Applies Conv2D, performs batch normalization, and applies Leaky ReLU
    activation to input `x`."""
    raise NotImplementedError('Your batches are not normal!!!')


# -------------------- #
# Data processing here #
# -------------------- #

# hyperparameters
num_epochs = 200
batch_size = 40
sound_shape = (512, 1378, 2)
latent_shape = (1, 1, 64)

# encoder network
input_sound = Input(shape=sound_shape, name='encoder_input')
x = Conv2D(16, kernel_size=(8, 16), strides=(4, 8))

print('Made it here in one piece!!!')
