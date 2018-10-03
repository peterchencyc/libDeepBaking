import numpy as np
from keras.models import load_model
from keras.preprocessing.image import array_to_img
from scipy import misc
from keras.preprocessing.image import array_to_img, img_to_array, load_img

import os
import re

import matplotlib.pyplot as plt
from PIL import Image
import tiling

import preprocessing
from numpy import linalg as LA


def predicting(model, test_or_train, root, input_root, mean, std):

    mean_l = mean[0:3]
    std_l = std[0:3]
    mean_r = mean[3:12]
    std_r = std[3:12]

    input_directory = input_root + '/' + test_or_train

    output_directory = root + 'prediction/' + test_or_train
    os.system('mkdir ' + output_directory)

    file_pattern = r'tile-w(\d+)z(\d+)s(\d+)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?).png'
    file_matcher = re.compile(file_pattern)
    lbls = np.array([])

    imgs_gt = np.array([])
    wmin = 50
    wrange = 70 - wmin
    zmin = 17
    zrange = 36 - zmin
    smin = 50
    srange = 275 - smin
    for file_name in os.listdir(input_directory):
        file_match = file_matcher.match(file_name)
        if file_match is None:
            continue

        w = float(file_match.group(1))
        z = float(file_match.group(2))
        s = float(file_match.group(3))
        number_rand_1 = float(file_match.group(4))
        number_rand_2 = float(file_match.group(5))
        rot = float(file_match.group(6))
        zoom = float(file_match.group(7))
        shear = float(file_match.group(8))
        shear_dir = float(file_match.group(9))
        h_mult = float(file_match.group(10))
        s_mult = float(file_match.group(11))
        v_mult = float(file_match.group(12))
        lbl = np.array([(w - wmin) / wrange - 0.5, (z - zmin) / zrange - 0.5, (s - smin) / srange -
                        0.5, number_rand_1, number_rand_2, rot, zoom, shear, shear_dir, h_mult, s_mult, v_mult])

        lbl = lbl.reshape((1,) + lbl.shape)

        lbl_l = lbl[:, 0:3]
        lbl_r = lbl[:, 3:12]

        # normalize before using
        lbl_l = (lbl_l - mean_l) / std_l
        lbl_r = lbl_r - mean_r

        lbl[:, 0:3] = lbl_l
        lbl[:, 3:12] = lbl_r

        if lbls.shape[0] == 0:
            lbls = lbl
        else:
            lbls = np.concatenate((lbls, lbl), axis=0)
        lbl = [lbl_l, lbl_r]

        input_image_path = input_directory + '/' + file_name
        img_gt = load_img(input_image_path)
        x_gt = img_to_array(img_gt)
        x_gt = x_gt / 255.0 * 2.0 - 1.0
        x_gt = x_gt.reshape((1,) + x_gt.shape)

        if imgs_gt.shape[0] == 0:
            imgs_gt = x_gt
        else:
            imgs_gt = np.concatenate((imgs_gt, x_gt), axis=0)

        imgs = model.predict(lbl)
        img = imgs[0, :, :, :]
        img = (img + 1.0) / 2.0 * 255.0

        output_image_path = output_directory + '/output_' + str(w) + '_' + str(z) + '_' + str(s) + '_' + str(number_rand_1) + '_' + str(number_rand_2) + '_' + str(
            rot) + '_' + str(zoom) + '_' + str(shear) + '_' + str(shear_dir) + '_' + str(h_mult) + '_' + str(s_mult) + '_' + str(v_mult) + '.png'
        img = np.uint8(img)
        im = Image.fromarray(img)
        im.putalpha(255)
        im.save(output_image_path)
    lbls_l = lbls[:, 0:3]
    lbls_r = lbls[:, 3:12]
    lbls = [lbls_l, lbls_r]
    scores = model.evaluate(lbls, imgs_gt, verbose=0)
    print test_or_train
    print 'model.metrics_names', model.metrics_names
    print 'scores', scores


def predicting_all():
    np.set_printoptions(threshold='nan')
    root = ''
    filepath = root + 'keras_model.dms'
    model = load_model(filepath)

    os.system('rm ' + root + 'prediction/*.png')
    os.system('rm ' + root + 'prediction/test/*.png')
    os.system('rm ' + root + 'prediction/train/*.png')

# The code is currently hard coded to train on w which can be easily
# changed by replacing the following directory
    w_z_s_c = 'w'
    input_root = 'data/original/lower_res_w_more/aug_1/'
    training_input_root = 'data/original/lower_res_w_more/aug_20/'

    mean = np.load(training_input_root + '/mean')
    std = np.load(training_input_root + '/std')
    predicting(model, 'train', root, input_root, mean, std)
    predicting(model, 'test', root, input_root, mean, std)

    loss = np.load('loss.npy')
    val_loss = np.load('val_loss.npy')

    plt.plot(loss, label="loss")
    plt.plot(val_loss, label="val_loss")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    ymin = 0
    ymax = 0.4
    plt.ylim(ymin, ymax)
    plt.savefig('prediction/loss')

    tiling.tiling(w_z_s_c=w_z_s_c)


def predict_and_save(model, w, z, s, number_rand_1, number_rand_2, rot, zoom, shear, shear_dir, h_mult, s_mult, v_mult, wmin, zmin, smin, wrange, zrange, srange, mean_l, std_l, mean_r, std_r, output_directory, count=0):
    lbl = np.array([(w - wmin) / wrange - 0.5, (z - zmin) / zrange - 0.5, (s - smin) / srange -
                    0.5, number_rand_1, number_rand_2, rot, zoom, shear, shear_dir, h_mult, s_mult, v_mult])
    lbl = lbl.reshape((1,) + lbl.shape)

    lbl_l = lbl[:, 0:3]
    lbl_r = lbl[:, 3:12]

    # normalize before using
    lbl_l = (lbl_l - mean_l) / std_l
    lbl_r = lbl_r - mean_r

    lbl[:, 0:3] = lbl_l
    lbl[:, 3:12] = lbl_r

    lbl = [lbl_l, lbl_r]

    imgs = model.predict(lbl)
    img = imgs[0, :, :, :]
    img = (img + 1.0) / 2.0 * 255.0

    output_image_path = output_directory + '/' + str(count) + '_' + 'output_' + str(w) + '_' + str(z) + '_' + str(s) + '_' + str(number_rand_1) + '_' + str(
        number_rand_2) + '_' + str(rot) + '_' + str(zoom) + '_' + str(shear) + '_' + str(shear_dir) + '_' + str(h_mult) + '_' + str(s_mult) + '_' + str(v_mult) + '.png'

    img = np.uint8(img)
    im = Image.fromarray(img)
    im.putalpha(255)
    im.save(output_image_path)


def predicting_inter():
    np.set_printoptions(threshold='nan')
    root = ''
    filepath = root + 'keras_model.dms'
    model = load_model(filepath)

    output_directory = root + 'prediction_inter'

    os.system('rm -rf ' + output_directory)
    os.system('mkdir ' + output_directory)

# The code is currently hard coded to train on w which can be easily
# changed by replacing the following directory
    input_root = 'data/original/lower_res_w_more/aug_1/'
    mean = np.load(input_root + '/mean')
    std = np.load(input_root + '/std')
    mean_l = mean[0:3]
    std_l = std[0:3]
    mean_r = mean[3:12]
    std_r = std[3:12]

    wmin = 50
    wrange = 70 - wmin
    zmin = 17
    zrange = 36 - zmin
    smin = 50
    srange = 275 - smin

    wmax = 70.0
    z = 30.0
    s = 175.0
    number_rand_1 = 0.0
    number_rand_2 = 0.0
    rot = 0.0
    zoom = 1.0
    shear = 0.0
    shear_dir = 0.0
    h_mult = 1.0
    s_mult = 1.0
    v_mult = 1.0

    for w in np.linspace(wmin, wmax, num=20).tolist():
        print 'w', w
        predict_and_save(model, w, z, s, number_rand_1, number_rand_2, rot, zoom, shear, shear_dir, h_mult, s_mult,
                         v_mult, wmin, zmin, smin, wrange, zrange, srange, mean_l, std_l, mean_r, std_r, output_directory)
