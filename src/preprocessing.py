import numpy as np
import os
import re
from scipy import misc
from keras.preprocessing.image import array_to_img, img_to_array, load_img, apply_transform
from numpy import linalg as LA

np.set_printoptions(threshold='nan')


def preProcessing(input_root, t_or_t, mean, std):

    input_directory = input_root + '/' + t_or_t

    file_pattern = r'tile-w(\d+)z(\d+)s(\d+)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?).png'
    file_matcher = re.compile(file_pattern)

    lbls = np.array([])
    imgs = np.array([])

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
        input_image_path = input_directory + '/' + file_name
        output_image_path = input_directory + '/' + 'output_' + file_name

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
        img = load_img(input_image_path)
        x = img_to_array(img)

        x = x / 255.0 * 2.0 - 1.0
        x = x.reshape((1,) + x.shape)

        if imgs.shape[0] == 0:
            imgs = x
        else:
            imgs = np.concatenate((imgs, x), axis=0)

        lbl = lbl.reshape((1,) + lbl.shape)

        if lbls.shape[0] == 0:
            lbls = lbl
        else:
            lbls = np.concatenate((lbls, lbl), axis=0)

    lbls_l = lbls[:, 0:3]
    lbls_r = lbls[:, 3:12]

    if t_or_t == 'train':
        mean_l = np.mean(lbls_l, axis=0)
        std_l = np.std(lbls_l, axis=0)
        mean[0:3] = mean_l
        std[0:3] = std_l
    elif t_or_t == 'test':
        mean_l = mean[0:3]
        std_l = std[0:3]
    else:
        print 'Does not support ' + t_or_t + ' as data input directory'
        exit()

    lbls_l = (lbls_l - mean_l) / std_l

    if t_or_t == 'train':
        mean_r = np.mean(lbls_r, axis=0)
        std_r = np.std(lbls_r, axis=0)
        mean[3:12] = mean_r
        std[3:12] = std_r
    elif t_or_t == 'test':
        mean_r = mean[3:12]
        std_r = std[3:12]
    else:
        print 'Does not support ' + t_or_t + ' as data input directory'
        exit()

    lbls_r = lbls_r - mean_r

    lbls[:, 0:3] = lbls_l
    lbls[:, 3:12] = lbls_r

    # save mean and std
    if t_or_t == 'train':
        save_path = input_root + '/mean'
        f1 = open(save_path, 'w')
        np.save(f1, mean)
        save_path = input_root + '/std'
        f2 = open(save_path, 'w')
        np.save(f2, std)

    return lbls, imgs
