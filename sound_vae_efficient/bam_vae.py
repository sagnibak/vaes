from conv import ShuffleNetStage, GlobalGammaPooling2D
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.datasets import cifar10
from keras.layers import BatchNormalization, Conv2D, Dense, GlobalAvgPool2D, Input, GlobalMaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import numpy as np


# hyperparameters
lr = 1e-3
decay = 2e-5
momentum = 0.9
batch_size = 32
epochs = 10

# TODO: Waiting for implementation of conv.SamplingAndFlow.
