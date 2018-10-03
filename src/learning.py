from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Conv2DTranspose, Reshape, ZeroPadding2D, advanced_activations, Conv2D, concatenate, Input
import preprocessing
from keras.utils import plot_model
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import argparse
import time
import numpy as np
import keras.backend as K


def build_generator():

    if K.image_dim_ordering() == "channels_first":
        bn_axis = 1
    else:
        bn_axis = -1

    filters = 128

    act = 'relu'

    kernel_initializer = 'he_normal'

    left_input = Input(shape=(3,))
    left = Dense(units=512, activation=act,
                 kernel_initializer=kernel_initializer)(left_input)
    right_input = Input(shape=(9,))
    right = Dense(units=512, activation=act,
                  kernel_initializer=kernel_initializer)(right_input)

    pre_model = concatenate([left, right])

    # now we define the rest of the model
    model = Sequential()

    model.add(Dense(units=1024, activation=act,
                    kernel_initializer=kernel_initializer, input_shape=(1024,)))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(Dense(units=1024, activation=act,
                    kernel_initializer=kernel_initializer))
    model.add(BatchNormalization(axis=bn_axis))

    model.add(Dense(units=filters * 4 * 4, activation=act,
                    kernel_initializer=kernel_initializer))
    model.add(Reshape((4, 4, filters)))
    model.add(BatchNormalization(axis=bn_axis))

    ks = 6

    model.add(Conv2DTranspose(int(filters), kernel_size=(ks, ks), strides=2,
                              padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(int(0.5 * filters), kernel_size=(ks, ks), strides=2,
                              padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(int(0.25 * filters), kernel_size=(ks, ks),
                              strides=2, padding='same', kernel_initializer='he_normal', use_bias=False))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(3, kernel_size=(ks, ks), strides=2, padding='same',
                              activation='tanh', kernel_initializer='he_normal', use_bias=False))

    final_model = model(pre_model)
    model = Model(inputs=[left_input, right_input], outputs=final_model)
    print("generator_model summary")
    model.summary()
    return model


parser = argparse.ArgumentParser(
    description='Deep Learn Laser Cooking model from cooked images.')
parser.add_argument('-l', metavar='Use Learned Model', type=int,
                    nargs=1, help='Indicate using learned model or not', required=True)
args = parser.parse_args()

use_learned_model = args.l[0]

if use_learned_model > 0:
    print('use_learned_model')
    root = ''
    filepath = root + 'keras_model'
    model = load_model(filepath)
else:
    print('NOT use_learned_model')

    model = build_generator()

    opt_dcgan = optimizers.Adam(
        lr=2E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    model.compile(loss='mae', metrics=['mae'],
                  optimizer=opt_dcgan)


# The code is currently hard coded to train on w which can be easily
# changed by replacing the following directory
root = 'data/original/lower_res_w_more/aug_20'

mean = np.zeros(12)
std = np.zeros(12)
lbls, imgs = preprocessing.preProcessing(root, 'train', mean, std)
lbls_l = lbls[:, 0:3]
lbls_r = lbls[:, 3:12]
lbls = [lbls_l, lbls_r]
lbls_test, imgs_test = preprocessing.preProcessing(root, 'test', mean, std)
lbls_test_l = lbls_test[:, 0:3]
lbls_test_r = lbls_test[:, 3:12]
lbls_test = [lbls_test_l, lbls_test_r]

start = time.time()
epochs = 10000
batch_size = 32

history = model.fit(lbls, imgs, validation_data=(
    lbls_test, imgs_test), epochs=epochs, batch_size=batch_size)
end = time.time()
print('training time per epoch', (end - start) / epochs)

root = ''
filepath = root + 'keras_model'
model.save(filepath)

loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])

np.save('loss.npy', loss)
np.save('val_loss.npy', val_loss)
