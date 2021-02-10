import json
import os
import pickle

import numpy as np
import pkg_resources
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from symspellpy import SymSpell, Verbosity
from tqdm import tqdm

cuda = True
device = 0

coco_inception_features_path = "../data/coco_train_v3_pytorch.pik"
coco_dataset_path = "../data/coco_dataset_full_rm_style.json"

model_path = "../models/"
test_model_fname = "img_to_txt_state.tar"

BATCH_SIZE = 32


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


def get_data(train=True):
    # feats = cPickle.load(open(coco_inception_features_path, "rb"), encoding="latin1")
    feats = pickle.load(open('../data/coco_train_v3.pik', "rb"), encoding="latin1")
    feats.update(pickle.load(open('../data/coco_val_ins.pik', "rb"), encoding="latin1"))

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
            sents = pickle.load(f)
    else:
        sents = correct_spell_for_dataset(data_file, sents)
    return final_feats, filenames, sents


def correct_spell_for_dataset(data_file, sents):
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
        pickle.dump(sents, f)
    return sents


def correct_spell(input_term, sym_spell):
    # lookup suggestions for single-word input strings
    # max edit distance per lookup
    # (max_edit_distance_lookup <= max_dictionary_edit_distance)
    suggestions = sym_spell.lookup(input_term, Verbosity.CLOSEST,
                                   max_edit_distance=2)
    # display suggestion term, term frequency, and edit distance
    return suggestions[0].term if len(suggestions) > 0 else input_term


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


class ImgEmb(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ImgEmb, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.emb_drop = nn.Dropout(0.2)
        self.mlp = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input):
        drop = self.emb_drop(input)
        mlp = self.mlp(drop)
        res = self.tanh(mlp)
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


def main():
    # ap = argparse.ArgumentParser()
    # ap.add_argument('--train', action='store_true')
    # ap.add_argument('--test_server', action='store_true')
    # ap.add_argument('--test_folder', help="Folder of images to run on")
    # ap.add_argument('--cpu', action='store_true')
    # args = ap.parse_args()
    # global cuda
    # if args.cpu:
    #     cuda = False
    # global test_model_fname
    # for i in tqdm(range(13)):
    #     test_model_fname = f'img_to_txt_state_{i}.tar'
    #     te = setup_test()
    #     test(setup_data=te)
    train()


# elif args.test_server:
#     r = setup_test()
#     run_server(r)
# else:
#     r = setup_test()
#     test(r, args.test_folder)


if __name__ == "__main__":
    main()
