import glob
import os
import pickle as cPickle

import keras
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from tqdm import tqdm

from img_to_text import coco_inception_features_path


def one_image():
    model = InceptionV3(weights='imagenet', include_top=False)

    img_path = "../train2014/COCO_train2014_000000001025.jpg"
    new_input = model.input
    hidden_layer = model.layers[-1].output
    model_ = keras.Model(new_input, hidden_layer)
    print(model_.summary())
    exit()

    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    reds = model_.predict(x)
    pred = np.mean(preds, axis=(1, 2))
    red = np.mean(reds, axis=(1, 2))
    print('')


def all_image():
    model = InceptionV3(weights='imagenet', include_top=False)
    img_path = "../val2014/"
    filenames = glob.glob(img_path + "*.jpg")
    batch_size = 512
    x = np.zeros((batch_size, 299, 299, 3), dtype=np.float32)
    cur_pos = 0
    all_out = {}
    cur_fnames = []
    c = 0
    for fname in tqdm(filenames, position=0):
        c += 1
        try:
            img = image.load_img(fname, target_size=(299, 299))
        except Exception as e:
            print(e)
            img = np.zeros((299, 299, 3), dtype=np.float32)
        x[cur_pos] = image.img_to_array(img)
        cur_fnames.append(os.path.basename(fname))

        if cur_pos == batch_size - 1:
            x = preprocess_input(x)
            preds = model(x)
            preds = np.mean(preds, axis=(1, 2))
            for k, v in zip(cur_fnames, preds):
                all_out[k] = v
            cur_fnames = []
            cur_pos = 0
            x[:, :, :, :] = 0.0
        else:
            cur_pos += 1
    if cur_pos != 0:
        x = preprocess_input(x)
        preds = model.predict(x[:cur_pos])
        preds = np.mean(preds, axis=(1, 2))
        for k, v in zip(cur_fnames, preds):
            all_out[k] = v
    cPickle.dump(all_out, open("../data/coco_val_ins.pik", "wb"), protocol=2)


all_image()
# if __name__ == '__main__':
#     with open('','wb') as f:
