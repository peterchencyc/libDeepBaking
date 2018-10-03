import numpy as np
import os
import re
from scipy import misc
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import platform


def tiling_flat(input_directory='prediction_inter'):
    root = ''
    imgs = np.array([])
    for file_name in os.listdir(input_directory):
        if not file_name.startswith('.') and file_name.endswith('.png'):
            print file_name
            input_image_path = input_directory + '/' + file_name
            img = load_img(input_image_path)
            x = img_to_array(img)
            x = x / 255.0 * 2.0 - 1.0
            x = x.reshape((1,) + x.shape)
            if imgs.shape[0] == 0:
                imgs = x
            else:
                imgs = np.concatenate((imgs, x), axis=0)

    img_tile = np.array([])
    for idx in range(imgs.shape[0]):
        if img_tile.shape[0] == 0:
            img_tile = imgs[idx, :, :, :]
        else:
            img_tile = np.concatenate((img_tile, imgs[idx, :, :, :]), axis=1)

    output_tile_path = root + 'tiles/' + str(imgs.shape[0]) + '.png'
    img_tile = (img_tile + 1) / 2.0 * 255.0
    img_tile = array_to_img(img_tile, scale=False)
    img_tile.putalpha(255)
    print 'output_tile_path', output_tile_path
    misc.imsave(output_tile_path, img_tile)


def tiling_square(root, water, z_list, s_list, gt, e, no_test=0, no_train=0):

    if gt > 0:
        file_pattern = r'tile-w' + \
            str(int(water)) + 'z(\d+)s(\d+)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?).png'
    else:
        file_pattern = r'output_' + \
            str(water) + '_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?)_([+-]?\d+(?:\.\d+)?).png'

    file_matcher = re.compile(file_pattern)

    test_or_train = 'train'
    input_directory = root + '/' + test_or_train

    for file_name in os.listdir(input_directory):

        file_match = file_matcher.match(file_name)
        if file_match is None:
            continue
        break
    input_image_path = input_directory + '/' + file_name
    img = Image.open(input_image_path)
    w = img.size[0]
    h = img.size[1]

    w_tile = len(z_list) * w
    h_tile = len(s_list) * h

    im = Image.new("RGB", (h_tile, w_tile), "white")
    im = np.array(im)

    if not no_train:
        for file_name in os.listdir(input_directory):
            file_match = file_matcher.match(file_name)

            if file_match is None:
                continue
            z = float(file_match.group(1))
            s = float(file_match.group(2))

            if z not in z_list or s not in s_list:
                continue

            i = len(z_list) - 1 - z_list.index(z)
            j = s_list.index(s)
            input_image_path = input_directory + '/' + file_name
            img = Image.open(input_image_path)
            img = np.array(img)
            im[w * i:(w * i + w), h * j:(h * j + h), :] = img[:, :, 0:3]

    test_or_train = 'test'
    input_directory = root + '/' + test_or_train
    if not no_test:
        for file_name in os.listdir(input_directory):
            file_match = file_matcher.match(file_name)
            if file_match is None:
                continue
            z = float(file_match.group(1))
            s = float(file_match.group(2))

            if z not in z_list or s not in s_list:
                continue

            i = len(z_list) - 1 - z_list.index(z)
            j = s_list.index(s)
            input_image_path = input_directory + '/' + file_name
            img = Image.open(input_image_path)
            img = np.array(img)
            im[w * i:(w * i + w), h * j:(h * j + h), :] = img[:, :, 0:3]

    ws = int((w_tile + h_tile / 2.0) * 0.1)
    im_large = Image.new("RGB", (h_tile + 2 * ws, w_tile + 2 * ws), "white")
    im_large = np.array(im_large)
    im_large[ws:(ws + w_tile), ws:(ws + h_tile), :] = im

    im_from_array = Image.fromarray(im_large)
    im_from_array.putalpha(255)

    draw = ImageDraw.Draw(im_from_array)

    if platform.system() == 'Darwin':
        font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 15)
    elif platform.system() == 'Linux':
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf", 15)

    x = int((h_tile + 2 * ws) * 0.5)
    y = int((w_tile + 2 * ws) * 0.5)
    draw.text((x, w_tile + 1.25 * ws), 'w = ' +
              str(water), (0, 0, 0), font=font)

    draw.text((x, 0.25 * ws), 's', (0, 0, 0), font=font)
    for z in z_list:
        i = len(z_list) - 1 - z_list.index(z)
        draw.text((0.5 * ws, ws + (i + 0.5) * w), str(z), (0, 0, 0), font=font)

    draw.text((0.25 * ws, y), 'z', (0, 0, 0), font=font)
    for s in s_list:
        j = s_list.index(s)
        draw.text((ws + (j + 0.5) * h, 0.5 * ws), str(s), (0, 0, 0), font=font)

    if gt > 0:
        output_image_path = root + '/' + 'tile_w_' + str(int(water)) + '_z_' + str(z_list[0]) + '_s_' + str(
            s_list[0]) + '_gt_' + str(e) + '_no-test_' + str(no_test) + '_no-train_' + str(no_train) + '.png'
    else:
        output_image_path = root + '/' + 'tile_w_' + str(int(water)) + '_z_' + str(z_list[0]) + '_s_' + str(
            s_list[0]) + '_result_' + str(e) + '_no-test_' + str(no_test) + '_no-train_' + str(no_train) + '.png'

    im_from_array.save(output_image_path)


def tiling(e=0, w_z_s_c='w'):

    input_result = 'prediction'

    if w_z_s_c == 'z':
        input_gt = 'data/original/lower_res_z_more/aug_1'
        water = 50.0
        z_list = [20, 26, 33]
        s_list = [150, 175, 200, 225, 250, 275]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)
        z_list = [23, 30, 36]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)

        water = 55.0
        z_list = [17, 23, 30, 36]
        s_list = [125, 150, 175, 200, 225, 250]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)
        z_list = [20, 26, 33]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)

        water = 60.0
        z_list = [17, 23, 30]
        s_list = [100, 125, 150, 175, 200, 225]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)
        z_list = [20, 26, 33]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)

        water = 65.0
        z_list = [17, 23, 30]
        s_list = [75, 100, 125, 150, 175, 200]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)
        z_list = [20, 26, 33]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)

        water = 70.0
        z_list = [17, 23, 30]
        s_list = [50, 75, 100, 125, 150, 175]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)
        z_list = [20, 26, 33]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)

    if w_z_s_c == 'w':
        input_gt = 'data/original/lower_res_w_more/aug_1'
        water = 50.0
        z_list = [20, 23, 26, 30, 33, 36]
        s_list = [150, 175, 200, 225, 250, 275]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)
        water = 55.0
        z_list = [17, 20, 23, 26, 30, 33, 36]
        s_list = [125, 150, 175, 200, 225, 250]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)
        water = 60.0
        z_list = [17, 20, 23, 26, 30, 33]
        s_list = [100, 125, 150, 175, 200, 225]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)
        water = 65.0
        z_list = [17, 20, 23, 26, 30, 33]
        s_list = [75, 100, 125, 150, 175, 200]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)
        water = 70.0
        z_list = [17, 20, 23, 26, 30, 33]
        s_list = [50, 75, 100, 125, 150, 175]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)

    if w_z_s_c == 's':
        input_gt = 'data/original/lower_res_s_more/aug_1'
        water = 50.0
        z_list = [20, 23, 26, 30, 33, 36]
        s_list = [150, 200, 250]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)
        s_list = [175, 225, 275]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)

        water = 55.0
        z_list = [17, 20, 23, 26, 30, 33, 36]
        s_list = [125, 175, 225]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)
        s_list = [150, 200, 250]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)

        water = 60.0
        z_list = [17, 20, 23, 26, 30, 33]
        s_list = [100, 150, 200]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)
        s_list = [125, 175, 225]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)

        water = 65.0
        z_list = [17, 20, 23, 26, 30, 33]
        s_list = [75, 125, 175]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)
        s_list = [100, 150, 200]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)

        water = 70.0
        z_list = [17, 20, 23, 26, 30, 33]
        s_list = [50, 100, 150]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)
        s_list = [75, 125, 175]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e)
        tiling_square(input_result, water, z_list, s_list, 0, e)

    if w_z_s_c == 'c':
        input_gt = 'data/original/lower_res_checker_board_more/aug_1'
        water = 50.0
        z_list = [20, 23, 26, 30, 33, 36]
        s_list = [150, 175, 200, 225, 250, 275]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e, no_test=1)
            tiling_square(input_gt, water, z_list, s_list, 1, e, no_train=1)
        tiling_square(input_result, water, z_list, s_list, 0, e, no_test=1)
        tiling_square(input_result, water, z_list, s_list, 0, e, no_train=1)
        water = 55.0
        z_list = [17, 20, 23, 26, 30, 33, 36]
        s_list = [125, 150, 175, 200, 225, 250]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e, no_test=1)
            tiling_square(input_gt, water, z_list, s_list, 1, e, no_train=1)
        tiling_square(input_result, water, z_list, s_list, 0, e, no_test=1)
        tiling_square(input_result, water, z_list, s_list, 0, e, no_train=1)
        water = 60.0
        z_list = [17, 20, 23, 26, 30, 33]
        s_list = [100, 125, 150, 175, 200, 225]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e, no_test=1)
            tiling_square(input_gt, water, z_list, s_list, 1, e, no_train=1)
        tiling_square(input_result, water, z_list, s_list, 0, e, no_test=1)
        tiling_square(input_result, water, z_list, s_list, 0, e, no_train=1)
        water = 65.0
        z_list = [17, 20, 23, 26, 30, 33]
        s_list = [75, 100, 125, 150, 175, 200]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e, no_test=1)
            tiling_square(input_gt, water, z_list, s_list, 1, e, no_train=1)
        tiling_square(input_result, water, z_list, s_list, 0, e, no_test=1)
        tiling_square(input_result, water, z_list, s_list, 0, e, no_train=1)
        water = 70.0
        z_list = [17, 20, 23, 26, 30, 33]
        s_list = [50, 75, 100, 125, 150, 175]
        if e <= 1:
            tiling_square(input_gt, water, z_list, s_list, 1, e, no_test=1)
            tiling_square(input_gt, water, z_list, s_list, 1, e, no_train=1)
        tiling_square(input_result, water, z_list, s_list, 0, e, no_test=1)
        tiling_square(input_result, water, z_list, s_list, 0, e, no_train=1)
