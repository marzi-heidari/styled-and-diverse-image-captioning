import json

import os

import pandas
import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from img_to_seq.utils import build_model, get_data, generate
from text_processing import untokenize, tokenize_text, pad_text
from seq_to_seq.utils import build_model as build_model_s2s, model_path, seq_to_seq_test_model_fname, COCO_STYLE, \
    BATCH_SIZE
import numpy as np

cuda = True
device = 0


class NopModule(torch.nn.Module):
    def __init__(self):
        super(NopModule, self).__init__()

    def forward(self, input):
        return input


def setup_test_s2s():
    if not cuda:
        loaded_state = torch.load(model_path + seq_to_seq_test_model_fname,
                                  map_location='cpu')
    else:
        loaded_state = torch.load(model_path + seq_to_seq_test_model_fname, map_location='cuda:0')

    enc_idx_to_word = loaded_state['enc_idx_to_word']
    enc_word_to_idx = loaded_state['enc_word_to_idx']
    enc_vocab_size = len(enc_idx_to_word)

    dec_idx_to_word = loaded_state['dec_idx_to_word']
    dec_word_to_idx = loaded_state['dec_word_to_idx']
    dec_vocab_size = len(dec_idx_to_word)

    enc, dec = build_model_s2s(enc_vocab_size, dec_vocab_size, loaded_state=loaded_state)

    return {'enc': enc, 'dec': dec, 'enc_idx_to_word': enc_idx_to_word, 'enc_word_to_idx': enc_word_to_idx,
            'enc_vocab_size': enc_vocab_size, 'dec_idx_to_word': dec_idx_to_word,
            'dec_word_to_idx': dec_word_to_idx, 'dec_vocab_size': dec_vocab_size}


def test_s2s(setup_data, input_seqs=None, test_style=COCO_STYLE):
    if input_seqs is None:
        _, input_rems_text = get_data(train=False, test_style=test_style)
    else:
        input_rems_text = input_seqs
        slen = len(input_seqs)
        for i in range(slen):
            input_rems_text[i].append(test_style)

    _, _, enc_tok_text, _ = tokenize_text(input_rems_text,
                                          idx_to_word=setup_data['enc_idx_to_word'],
                                          word_to_idx=setup_data['enc_word_to_idx'])
    enc_padded_text = pad_text(enc_tok_text)

    dlen = enc_padded_text.shape[0]
    num_batch = dlen // BATCH_SIZE
    if dlen % BATCH_SIZE != 0:
        num_batch += 1
    res = []
    for i in range(num_batch):
        dec_tensor = generate(setup_data['enc'], setup_data['dec'],
                              enc_padded_text[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])
        res.append(dec_tensor)

    all_text = []
    if len(res) > 0:
        res = np.concatenate(res, axis=0)
    for row in res:
        utok = untokenize(row, setup_data['dec_idx_to_word'], to_text=True)
        all_text.append(utok)
    return all_text

    # for i in xrange(100):
    #    print "IN :", untokenize(enc_padded_text[i], enc_idx_to_word, to_text=True)
    #    print "GEN:", untokenize(dec_tensor[i], dec_idx_to_word, to_text=True), "\n"


def setup_test(with_cnn=False):
    cnn, trans = get_cnn()
    if not cuda:
        loaded_state = torch.load(model_path + test_model_fname,
                                  map_location='cpu')
    else:
        loaded_state = torch.load(model_path + test_model_fname)
    dec_vocab_size = len(loaded_state['dec_idx_to_word'])
    enc, dec = build_model(dec_vocab_size, loaded_state=loaded_state)

    s2s_data = setup_test_s2s()
    return {'cnn': cnn, 'trans': trans, 'enc': enc, 'dec': dec,
            'loaded_state': loaded_state, 's2s_data': s2s_data}


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

        text = test_s2s(s2s_data, untok)

        for i in range(len(text)):
            filenames[k] = filenames[k].replace('COCO_val2014_', '')
            filenames[k] = filenames[k].replace('.jpg', '')
            j = {"image_id": int(filenames[k]), "caption": text[i], "words": ' '.join(untok[i])}
            id_captions.append(j)
            k += 1

        all_text.extend(text)
        with open('results/captions_val2014_' + test_model_fname + seq_to_seq_test_model_fname + '_results.json',
                  'w') as outfile:
            json.dump(id_captions, outfile)

        pandas.DataFrame(id_captions).to_csv(
            'results/captions_val2014_' + test_model_fname + seq_to_seq_test_model_fname + '_results.csv',
            index=False)
    return all_text


if __name__ == '__main__':
    global test_model_fname
    for i in tqdm(range(13)):
        test_model_fname = f'img_to_txt_state_{i}.tar'
        te = setup_test()
        test(setup_data=te)
