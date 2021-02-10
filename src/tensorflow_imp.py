import collections

import tensorflow as tf


# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
# Scikit-learn includes many helpful utilities


def main():
    # Download caption annotation files
    annotation_folder = '../annotations_trainval2014/annotations/'
    annotation_file = '../annotations_trainval2014/annotations/captions_train2014.json'

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    PATH = '../train2014'
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
        image_path_to_caption[image_path].append(caption)

    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)

    # Select the first 6000 image_paths from the shuffled set.
    # Approximately each image id has 5 captions associated with it, so that will
    # lead to 30,000 examples.
    train_image_paths = image_paths[:6000]
    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))

    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


import json
import os
import random

import numpy as np
import pandas
import pickle5 as cPickle
import pkg_resources
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from symspellpy import SymSpell, Verbosity
from torch import optim
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm

import seq2seq_pytorch as s2s
from text_processing import tokenize_text, untokenize, pad_text

cuda = False
device = 0

coco_inception_features_path = "../data/coco_train_v3_pytorch.pik"
coco_dataset_path = "../data/coco_dataset_full_rm_style.json"

model_path = "../models/"
test_model_fname = "img_to_txt_state.tar"
epoch_to_save_path = lambda epoch: model_path + "img_to_txt_state_%d.tar" % int(epoch)


BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 300
units = 512
vocab_size = 10002
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048




class NopModule(torch.nn.Module):
    def __init__(self):
        super(NopModule, self).__init__()

    def forward(self, input):
        return input


def get_cnn():
    inception = models.inception_v3(pretrained=True)
    inception.fc = NopModule()
    if cuda:
        inception = inception.cuda(device=device)
    inception.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize,
    ])
    return inception, trans


def has_image_ext(path):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    ext = os.path.splitext(path)[1]
    if ext.lower() in IMG_EXTENSIONS:
        return True
    return False


def list_image_folder(root):
    images = []
    dir = os.path.expanduser(root)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if os.path.isdir(d):
            continue
        if has_image_ext(d):
            images.append(d)
    return images


def safe_pil_loader(path, from_memory=False):
    try:
        if from_memory:
            img = Image.open(path)
            res = img.convert('RGB')
        else:
            with open(path, 'rb') as f:
                img = Image.open(f)
                res = img.convert('RGB')
    except:
        res = Image.new('RGB', (299, 299), color=0)
    return res


class ImageTestFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.loader = safe_pil_loader
        self.transform = transform

        self.samples = list_image_folder(root)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        return sample, path

    def __len__(self):
        return len(self.samples)


# load images provided across the network
class ImageNetLoader(torch.utils.data.Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.loader = safe_pil_loader
        self.transform = transform

    def __getitem__(self, index):
        sample = self.loader(self.images[index], from_memory=True)
        sample = self.transform(sample)
        return sample, ""

    def __len__(self):
        return len(self.images)


def get_image_reader(dirpath, transform, batch_size, workers=4):
    image_reader = torch.utils.data.DataLoader(
        ImageTestFolder(dirpath, transform),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    return image_reader


def get_data(train=True):
    # feats = cPickle.load(open(coco_inception_features_path, "rb"), encoding="latin1")
    feats = cPickle.load(open('../data/coco_train_v3.pik', "rb"), encoding="latin1")
    feats.update(cPickle.load(open('../data/coco_val_ins.pik', "rb"), encoding="latin1"))

    sents = []
    final_feats = []
    filenames = []
    js = json.load(open(coco_dataset_path, "r"))
    for i, img in enumerate(js["images"]):
        if train and img["extrasplit"] == "val":
            continue
        if (not train) and img["extrasplit"] != "val":
            continue
        if img["filename"] not in feats:
            continue
        if train:
            for sen in img["sentences"]:
                sents.append(sen["rm_style_tokens"])
                final_feats.append(feats[img["filename"]])
                filenames.append(img["filename"])
        else:
            sents.append(img["sentences"][0]["rm_style_tokens"])
            final_feats.append(feats[img["filename"]])
            filenames.append(img["filename"])

    final_feats = np.array(final_feats)
    data_file = 'cleaned_sents_train.pkl' if train is True else 'cleaned_test_train.pkl'
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            sents = cPickle.load(f)
    else:
        m = []
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=3)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt")
        # term_index is the column of the term and count_index is the
        # column of the term frequency
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        for i in tqdm(sents, position=0):
            l = []
            for j in range(len(i)):
                t = correct_spell(
                    i[j].replace('NOUNNOUNNOUN', '').replace("PARTPARTPART", "").replace("FRAMENET", "").replace(
                        "ADJADJADJ", "").replace('INTJINTJINTJ', '').lower(), sym_spell)
                l.append(t)
            m.append(l)
        sents = m

        with open(data_file, 'wb') as f:
            cPickle.dump(sents, f)
    return final_feats, filenames, sents


def correct_spell(input_term, sym_spell):
    # lookup suggestions for single-word input strings
    # max edit distance per lookup
    # (max_edit_distance_lookup <= max_dictionary_edit_distance)
    suggestions = sym_spell.lookup(input_term, Verbosity.CLOSEST,
                                   max_edit_distance=2)
    # display suggestion term, term frequency, and edit distance
    return suggestions[0].term if len(suggestions) > 0 else input_term


class ImgEmb(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ImgEmb, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.emb_drop = nn.Dropout(0.2)
        self.mlp = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input):
        res = self.tanh(self.mlp(self.emb_drop(input)))
        return res


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, out_bias=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.emb_drop = nn.Dropout(0.2)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru_drop = nn.Dropout(0.2)
        self.mlp = nn.Linear(hidden_size, output_size)
        if out_bias is not None:
            out_bias_tensor = torch.tensor(out_bias, requires_grad=False)
            self.mlp.bias.data[:] = out_bias_tensor
        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden_in):
        emb = self.embedding(input)
        emb = self.emb_drop(emb)
        out, hidden = self.gru(emb, hidden_in)
        out = self.mlp(self.gru_drop(out))
        out = self.logsoftmax(out)
        return out, hidden


def build_model(dec_vocab_size, dec_bias=None, img_feat_size=2048,
                hid_size=512, loaded_state=None):
    enc = ImgEmb(img_feat_size, hid_size)
    dec = Decoder(dec_vocab_size, hid_size, dec_vocab_size, dec_bias)
    if loaded_state is not None:
        enc.load_state_dict(loaded_state['enc'])
        dec.load_state_dict(loaded_state['dec'])
    if cuda:
        enc = enc.cuda(device=device)
        dec = dec.cuda(device=device)
    return enc, dec


def build_trainers(enc, dec, loaded_state=None):
    learning_rate = 0.001
    lossfunc = nn.NLLLoss(ignore_index=0)
    enc_optim = optim.Adam(enc.parameters(), lr=learning_rate, weight_decay=1e-6)
    dec_optim = optim.Adam(dec.parameters(), lr=learning_rate, weight_decay=1e-6)
    if loaded_state is not None:
        enc_optim.load_state_dict(load_state['enc_optim'])
        dec_optim.load_state_dict(load_state['dec_optim'])
    return enc_optim, dec_optim, lossfunc


def save_state(enc, dec, enc_optim, dec_optim, dec_idx_to_word, dec_word_to_idx, epoch):
    state = {'enc': enc.state_dict(), 'dec': dec.state_dict(),
             'enc_optim': enc_optim.state_dict(), 'dec_optim': dec_optim.state_dict(),
             'dec_idx_to_word': dec_idx_to_word, 'dec_word_to_idx': dec_word_to_idx}
    torch.save(state, epoch_to_save_path(epoch))


def generate(enc, dec, feats, L=20):
    enc.eval()
    dec.eval()
    with torch.no_grad():
        hid_enc = enc(feats).unsqueeze(0)

        # run the decoder step by step
        dec_tensor = torch.zeros(feats.shape[0], L + 1, dtype=torch.long)
        if cuda:
            dec_tensor = dec_tensor.cuda(device=device)
        last_enc = hid_enc
        for i in range(L):
            out_dec, hid_dec = dec.forward(dec_tensor[:, i].unsqueeze(1), last_enc)
            if out_dec.shape[0] <= 1:
                break
            chosen = torch.argmax(out_dec[:, 0], dim=1)
            dec_tensor[:, i + 1] = chosen
            last_enc = hid_dec

    return dec_tensor.data.cpu().numpy()


def setup_test(with_cnn=False):
    cnn, trans = get_cnn()
    if not cuda:
        loaded_state = torch.load(model_path + test_model_fname,
                                  map_location='cpu')
    else:
        loaded_state = torch.load(model_path + test_model_fname)
    dec_vocab_size = len(loaded_state['dec_idx_to_word'])
    enc, dec = build_model(dec_vocab_size, loaded_state=loaded_state)

    s2s.cuda = cuda
    s2s_data = s2s.setup_s2s_test()
    return {'cnn': cnn, 'trans': trans, 'enc': enc, 'dec': dec,
            'loaded_state': loaded_state, 's2s_data': s2s_data}


class TestIterator:
    def __init__(self, feats, text, bs=BATCH_SIZE):
        self.feats = feats
        self.text = text
        self.bs = bs
        self.num_batch = feats.shape[0] / bs
        if feats.shape[0] % bs != 0:
            self.num_batch += 1
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.num_batch:
            raise StopIteration()

        s = self.i * self.bs
        e = min((self.i + 1) * self.bs, self.feats.shape[0])
        self.i += 1
        return self.feats[s:e], self.text[s:e]


def test(setup_data, test_folder=None, test_images=None):
    enc = setup_data['enc']
    dec = setup_data['dec']
    cnn = setup_data['cnn']
    trans = setup_data['trans']
    loaded_state = setup_data['loaded_state']
    s2s_data = setup_data['s2s_data']
    k = 0

    dec_vocab_size = len(loaded_state['dec_idx_to_word'])
    id_captions = []
    json_data = json.dumps(id_captions)

    if test_folder is not None:
        # load images from folder
        img_reader = get_image_reader(test_folder, trans, BATCH_SIZE)
        using_images = True
    elif test_images is not None:
        # load images from memory
        img_reader = torch.utils.data.DataLoader(
            ImageNetLoader(test_images, trans),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=1, pin_memory=True)
        using_images = True
    else:
        # load precomputed image features from dataset
        feats, filenames, sents = get_data(train=False)
        feats_tensor = torch.tensor(feats, requires_grad=False)
        if cuda:
            feats_tensor = feats_tensor.cuda(device=device)
        img_reader = TestIterator(feats_tensor, sents)
        using_images = False

    all_text = []
    for input, text_data in img_reader:

        if using_images:
            if cuda:
                input = input.cuda(device=device)
            with torch.no_grad():
                batch_feats_tensor = cnn(input)
        else:
            batch_feats_tensor = input

        dec_tensor = generate(enc, dec, batch_feats_tensor)

        untok = []
        for i in range(dec_tensor.shape[0]):
            untok.append(untokenize(dec_tensor[i],
                                    loaded_state['dec_idx_to_word'],
                                    to_text=False))

        text = s2s.test_s2s(s2s_data, untok)

        for i in range(len(text)):
            filenames[k] = filenames[k].replace('COCO_val2014_', '')
            filenames[k] = filenames[k].replace('.jpg', '')
            j = {"image_id": int(filenames[k]), "caption": text[i], "words": ' '.join(untok[i])}
            id_captions.append(j)
            k += 1

        all_text.extend(text)
        with open('results/captions_val2014_' + test_model_fname + s2s.seq_to_seq_test_model_fname + '_results.json',
                  'w') as outfile:
            json.dump(id_captions, outfile)

        pandas.DataFrame(id_captions).to_csv(
            'results/captions_val2014_' + test_model_fname + s2s.seq_to_seq_test_model_fname + '_results.csv',
            index=False)
    return all_text

def train():
    feats, filenames, sents = get_data(train=True)

    dec_idx_to_word, dec_word_to_idx, dec_tok_text, dec_bias = tokenize_text(sents)
    dec_padded_text = pad_text(dec_tok_text)
    dec_vocab_size = len(dec_idx_to_word)

    enc, dec = build_model(dec_vocab_size, dec_bias)
    enc_optim, dec_optim, lossfunc = build_trainers(enc, dec)

    feats_tensor = torch.tensor(feats, requires_grad=False)
    dec_text_tensor = torch.tensor(dec_padded_text, requires_grad=False)
    if cuda:
        feats_tensor = feats_tensor.cuda(device=device)
        dec_text_tensor = dec_text_tensor.cuda(device=device)

    num_batches = feats.shape[0] // BATCH_SIZE

    sm_loss = None
    enc.train()
    dec.train()
    for epoch in range(0, 13):
        print("Starting New Epoch: %d" % epoch)

        order = np.arange(feats.shape[0])
        np.random.shuffle(order)
        del feats_tensor, dec_text_tensor
        if cuda:
            torch.cuda.empty_cache()
        feats_tensor = torch.tensor(feats[order], requires_grad=False)
        dec_text_tensor = torch.tensor(dec_padded_text[order], requires_grad=False)
        if cuda:
            feats_tensor = feats_tensor.cuda(device=device)
            dec_text_tensor = dec_text_tensor.cuda(device=device)

        for i in range(num_batches):
            s = i * BATCH_SIZE
            e = (i + 1) * BATCH_SIZE

            enc.zero_grad()
            dec.zero_grad()

            hid_enc = enc.forward(feats_tensor[s:e]).unsqueeze(0)
            out_dec, hid_dec = dec.forward(dec_text_tensor[s:e, :-1], hid_enc)

            out_perm = out_dec.permute(0, 2, 1)
            loss = lossfunc(out_perm, dec_text_tensor[s:e, 1:])

            if sm_loss is None:
                sm_loss = loss.data
            else:
                sm_loss = sm_loss * 0.95 + 0.05 * loss.data

            loss.backward()
            clip_grad_value_(dec_optim.param_groups[0]['params'], 5.0)

            enc_optim.step()
            dec_optim.step()

            if i % 100 == 0:
                print("Epoch: %.3f" % (i / float(num_batches) + epoch,), "Loss:", sm_loss)
                print("GEN:", untokenize(torch.argmax(out_dec, dim=2)[0, :], dec_idx_to_word))
                print("GT:", untokenize(dec_text_tensor[s, :], dec_idx_to_word))
                print("--------------")

        save_state(enc, dec, enc_optim, dec_optim, dec_idx_to_word, dec_word_to_idx, epoch)


if __name__ == '__main__':
    train()
