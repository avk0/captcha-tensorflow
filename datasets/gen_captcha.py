# -*- coding:utf-8 -*-
import argparse
import json
import string
from math import factorial
import os
import shutil
import uuid
from captcha.image import ImageCaptcha

import itertools


FLAGS = None
META_FILENAME = 'meta.json'
FONTS = ['./fonts/arial.ttf', './fonts/smartie.ttf']  # if not set, default will be used
FONT_SIZES = [62]  # if not set, default will be used
WIDTH = 210
HEIGHT = 60
FORMAT = 'jpeg'


def get_choices():
    choices = [
        (FLAGS.digit, map(str, range(10))),
        (FLAGS.lower, string.ascii_lowercase),
        (FLAGS.upper, string.ascii_uppercase),
        ]
    return tuple([i for is_selected, subset in choices for i in subset if is_selected])


def _gen_captcha(img_dir, num_per_image, n, e, fonts, font_sizes, width, height, choices):

    def number_of_permutations(n, r):
        return int(factorial(n) / factorial(n - r))

    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image = ImageCaptcha(width=width, height=height, fonts=fonts, font_sizes=font_sizes)

    # check if number of imgs is less than one epoch
    permutations_n = number_of_permutations(len(choices), num_per_image)
    print('Number of permutations %s, number needed %s' % (permutations_n, n))
    if 0 < n < permutations_n:
        e = 1
        coeff = permutations_n // n
    else:
        coeff = 1

    print('generating %s epoches of captchas in %s' % (e, img_dir))
    print('images to be genetated:', e * permutations_n // coeff)
    count = 0
    for _ in range(e):
        for i in itertools.permutations(choices, num_per_image):
            count += 1
            if count % coeff != 0:
                continue
            captcha = ''.join(i)
            fn = os.path.join(img_dir, '%s_%s.%s' % (captcha, uuid.uuid4(), FORMAT))
            print(fn)
            image.write(captcha, fn, format=FORMAT)


def build_file_path(x):
    return os.path.join(FLAGS.data_dir, 'char-%s-epoch-%s' % (FLAGS.npi, FLAGS.n), x)


def gen_dataset():
    n_captchas = FLAGS.n
    n_epoch = FLAGS.e
    num_per_image = FLAGS.npi
    test_ratio = FLAGS.t

    choices = get_choices()

    # meta info
    meta = {
        'num_per_image': num_per_image,
        'label_size': len(choices),
        'label_choices': ''.join(choices),
        'n_epoch': n_epoch,
        'width': WIDTH,
        'height': HEIGHT,
        'fonts': FONTS,
        'font_sizes': FONT_SIZES
    }

    print('%s choices: %s' % (len(choices), ''.join(choices) or None))

    _gen_captcha(build_file_path('train'), num_per_image, n_captchas, n_epoch,
                 FONTS, FONT_SIZES, WIDTH, HEIGHT, choices=choices)
    _gen_captcha(build_file_path('test'), num_per_image, n_captchas, max(1, int(n_epoch * test_ratio)),
                 FONTS, FONT_SIZES, WIDTH, HEIGHT, choices=choices)

    meta_filename = build_file_path(META_FILENAME)
    with open(meta_filename, 'w') as f:
        json.dump(meta, f, indent=4)
    print('write meta info in %s' % meta_filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        default=0,
        type=int,
        help='number of captchas to generate.')
    parser.add_argument(
        '-e',
        default=1,
        type=int,
        help='epoch number of character permutations.')

    parser.add_argument(
        '-t',
        default=0.2,
        type=float,
        help='ratio of test dataset.')

    parser.add_argument(
        '-d', '--digit',
        action='store_true',
        help='use digits in dataset.')
    parser.add_argument(
        '-l', '--lower',
        action='store_true',
        help='use lowercase in dataset.')
    parser.add_argument(
        '-u', '--upper',
        action='store_true',
        help='use uppercase in dataset.')
    parser.add_argument(
        '--npi',
        default=1,
        type=int,
        help='number of characters per image.')
    parser.add_argument(
        '--data_dir',
        default='./images',
        type=str,
        help='where data will be saved.')

    FLAGS, unparsed = parser.parse_known_args()

    gen_dataset()
