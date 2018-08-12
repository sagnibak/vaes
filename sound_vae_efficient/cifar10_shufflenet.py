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

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 127.5 - 1.
x_test = x_test / 127.5 - 1.
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


in_img = Input(shape=(32, 32, 3))
with K.name_scope('Simple_Convolution'):
    x = Conv2D(24, (3, 3), strides=(1, 1), padding='same')(in_img)  # for CIFAR-10
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
x = ShuffleNetStage(3, 4, 24, 96, bottleneck_ratio=3)(x)
x = ShuffleNetStage(3, 4, 96, 144, bottleneck_ratio=4)(x)
x = ShuffleNetStage(4, 4, 144, 192, bottleneck_ratio=3)(x)
x = ShuffleNetStage(4, 4, 192, 384, bottleneck_ratio=4)(x)
x = GlobalGammaPooling2D(gamma=0.5, mode='harmonic')(x)
x = Dense(10, activation='softmax')(x)

model = Model(in_img, x)
model.summary()

sgd = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=True)

tensorboard = TensorBoard(log_dir='../cifar10_logs1/run12')
modelcheckpoint = ModelCheckpoint('cifar10_shufflenet_models/model_12_{epoch:02d}_{val_loss:.2f}.h5',
                                  save_best_only=True)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[tensorboard, modelcheckpoint],
          validation_split=0.2,
          shuffle=True,
          verbose=1)
