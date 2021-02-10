import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel

from img_to_seq.utils import correct_spell_for_dataset
from text_processing import tokenize_text, Toks

cuda = True
device = 0
ROM_STYLE = "romancetoken"
COCO_STYLE = "mscocotoken"
model_path = "../models/"

epoch_to_save_path = lambda epoch: model_path + "seq_to_txt_state_%d.tar" % int(epoch)
rom_train_path = "../data/romance_wo_style.json"
coco_train_path = "../data/coco_dataset_full_rm_style.json"

seq_to_seq_test_model_fname = f"seq_to_txt_state_{11}.tar"

BATCH_SIZE = 32


def get_data(train=True, maxlines=-1, test_style=COCO_STYLE):
    input_text = []
    input_rems_text = []

    if train:
        js = json.load(open(rom_train_path, "r"))
        c = 0
        for line in js:
            sent = line[0]
            input_text.append(sent)
            rem_style = line[1]
            input_rems_text.append(rem_style + [ROM_STYLE])
            c += 1
            if 0 < maxlines == c:
                break

    c = 0
    js = json.load(open(coco_train_path, "r"))
    for i, img in enumerate(js["images"]):
        if train and img["extrasplit"] == "val":
            continue
        if (not train) and img["extrasplit"] == "train":
            continue
        if 0 < maxlines == c:
            break
        for sen in img["sentences"]:
            if train:
                input_rems_text.append(sen["rm_style_tokens"] + [COCO_STYLE])
            else:
                input_rems_text.append(sen["rm_style_tokens"] + [test_style])
            c += 1
            if 0 < maxlines == c:
                break

            input_text.append(sen["tokens"])
    data_file = 'cleaned_input_rems_text_train.pkl' if train is True else 'cleaned_input_rems_text_test.pkl'
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            input_rems_text = pickle.load(f)
    else:
        input_rems_text = correct_spell_for_dataset(data_file, input_rems_text)
    data_input_file = 'cleaned_input_text_train.pkl' if train is True else 'cleaned_input_text_test.pkl'
    if os.path.exists(data_input_file):
        with open(data_input_file, 'rb') as f:
            input_text = pickle.load(f)
    else:
        input_text = correct_spell_for_dataset(data_input_file, input_text)

    return input_text, input_rems_text


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, words=None):
        super(Encoder, self).__init__()
        assert hidden_size % 2 == 0

        self.hidden_size = hidden_size
        self.input_size = input_size

        self.hidden_init_tensor = torch.zeros(2, 1, self.hidden_size // 2, requires_grad=True)
        nn.init.normal_(self.hidden_init_tensor, mean=0, std=0.05)
        self.hidden_init = torch.nn.Parameter(self.hidden_init_tensor, requires_grad=True)
        with open('weights_for_seq_to_seq.pkl', 'rb') as f:
            weights_matrix = pickle.load(f)
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.embedding.requires_grad = False
        self.bn = nn.BatchNorm1d(20)
        # self.embedding = nn.Embedding(input_size, hidden_size)
        self.trans = nn.Linear(in_features=300, out_features=512)
        self.emb_drop = nn.Dropout(0.2)
        # self.gru=RNNModel(rnn_type='GRU',ntoken=words, ninp=512,nhid=512/2,nlayers=1)
        self.gru = nn.GRU(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.gru_out_drop = nn.Dropout(0.2)
        self.gru_hid_drop = nn.Dropout(0.3)

    def forward(self, input, hidden, lengths):
        # emb = self.emb_drop(self.embedding(input))
        # emb = self.embedding(input)
        emb = embedded_dropout(self.embedding, input, dropout=0.2 if self.training else 0)
        batch = self.bn(emb)
        batch = self.trans(batch)

        # pp = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True)
        pp = torch.nn.utils.rnn.pack_padded_sequence(batch, lengths, batch_first=True)
        out, hidden = self.gru(pp, hidden)
        out = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]
        out = self.gru_out_drop(out)
        hidden = self.gru_hid_drop(hidden)
        return out, hidden

    def initHidden(self, bs):
        # return self.gru.init_hidden(bs)
        return self.hidden_init.expand(2, bs, self.hidden_size // 2).contiguous()


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = torch.nn.functional.embedding(words, masked_embed_weight,
                                      padding_idx, embed.max_norm, embed.norm_type,
                                      embed.scale_grad_by_freq,
                                      embed.sparse
                                      )
    return X


class DecoderAttn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, out_bias, words=None):
        super(DecoderAttn, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.trans = nn.Linear(in_features=300, out_features=512)
        self.emb_drop = nn.Dropout(0.2)
        # with open('weights_for_seq_to_seq_decoder.pkl', 'rb') as f:
        #     weights_matrix = pickle.load(f)
        # self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        # self.embedding.requires_grad = False
        self.embedding = nn.Embedding(input_size, 300)
        # self.gru=RNNModel(rnn_type='GRU',ntoken=words, ninp=hidden_size,nhid=hidden_size,nlayers=1,bi=False)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru_drop = nn.Dropout(0.2)
        self.mlp = nn.Linear(hidden_size * 2, output_size)
        if out_bias is not None:
            out_bias_tensor = torch.tensor(out_bias, requires_grad=False)
            self.mlp.bias.data[:] = out_bias_tensor
        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.att_mlp = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_softmax = nn.Softmax(dim=2)

    def forward(self, input, hidden, encoder_outs):
        emb = embedded_dropout(self.embedding, input, dropout=0.2 if self.training else 0)
        # emb = self.embedding(input)
        emb = self.trans(emb)
        out, hidden = self.gru(emb, hidden)

        out_proj = self.att_mlp(out)
        enc_out_perm = encoder_outs.permute(0, 2, 1)
        e_exp = torch.bmm(out_proj, enc_out_perm)
        attn = self.attn_softmax(e_exp)

        ctx = torch.bmm(attn, encoder_outs)

        full_ctx = torch.cat([self.gru_drop(out), ctx], dim=2)

        out = self.mlp(full_ctx)
        out = self.logsoftmax(out)
        return out, hidden, attn


def build_model(enc_vocab_size, dec_vocab_size, dec_bias=None, hid_size=512, loaded_state=None):
    enc = Encoder(enc_vocab_size, hid_size)
    dec = DecoderAttn(dec_vocab_size, hid_size, dec_vocab_size, dec_bias)
    if loaded_state is not None:
        enc.load_state_dict(loaded_state['enc'])
        dec.load_state_dict(loaded_state['dec'])
    if cuda:
        enc = enc.cuda(device=device)
        dec = dec.cuda(device=device)
    return enc, dec


def make_packpadded(s, e, enc_padded_text, dec_text_tensor=None):
    text = enc_padded_text[s:e]
    lengths = np.count_nonzero(text, axis=1)
    order = np.argsort(-lengths)
    new_text = text[order]
    new_enc = torch.tensor(new_text)
    if cuda:
        new_enc = new_enc.cuda(device=device)

    if dec_text_tensor is not None:
        new_dec = dec_text_tensor[s:e][order].contiguous()
        leng = torch.tensor(lengths[order])
        if cuda:
            leng.cuda(device=device)
        return order, new_enc, new_dec, leng
    else:
        leng = torch.tensor(lengths[order])
        if cuda:
            leng.cuda(device=device)
        return order, new_enc, leng


def generate(enc, dec, enc_padded_text, L=20):
    enc.eval()
    dec.eval()
    with torch.no_grad():
        # run the encoder
        order, enc_pp, enc_lengths = make_packpadded(0, enc_padded_text.shape[0], enc_padded_text)
        hid = enc.initHidden(enc_padded_text.shape[0])
        out_enc, hid_enc = enc(enc_pp, hid, enc_lengths)

        hid_enc = torch.cat([hid_enc[0, :, :], hid_enc[1, :, :]], dim=1).unsqueeze(0)

        # run the decoder step by step
        dec_tensor = torch.ones((enc_padded_text.shape[0]), L + 1, dtype=torch.long) * Toks.SOS
        if cuda:
            dec_tensor = dec_tensor.cuda(device=device)
        last_enc = hid_enc
        for i in range(L):
            out_dec, hid_dec, attn = dec.forward(dec_tensor[:, i].unsqueeze(1), last_enc, out_enc)
            out_dec[:, 0, Toks.UNK] = -np.inf  # ignore unknowns
            # out_dec[torch.arange(dec_tensor.shape[0], dtype=torch.long), 0, dec_tensor[:, i]] = -np.inf
            chosen = torch.argmax(out_dec[:, 0], dim=1)
            dec_tensor[:, i + 1] = chosen
            last_enc = hid_dec

    return dec_tensor.data.cpu().numpy()[np.argsort(order)]


def extract_glove():
    words = []
    idx = 0
    word2idx = {}
    vectors = []
    with open('../glove.42B.300d.txt', 'r') as f:
        for l in f:
            line = l.split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array([float(val) for val in line[1:]])
            vectors.append(vect)
        vectors = np.array(vectors)
    glove = {w: vectors[word2idx[w]] for w in words}
    return glove


def get_emb_weights_for_seq_to_seq(glove, encoder=True):
    input_text, input_rems_text = get_data(train=True)
    if encoder:
        enc_idx_to_word, word_to_idx, enc_tok_text, _ = tokenize_text(input_rems_text)
    else:
        dec_idx_to_word, word_to_idx, dec_tok_text, dec_bias = tokenize_text(input_text, lower_case=True,
                                                                             vsize=20000)
    target_words = word_to_idx.keys()
    matrix_len = len(target_words)
    weights_matrix = np.zeros((matrix_len, 300))
    words_found = 0

    for i, word in enumerate(target_words):
        try:
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(300,))
    return weights_matrix


def get_bert_word_emb(text_array):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load pre-trained model (weights)
    BertModel = BertModel.from_pretrained('bert-base-uncased').to(device)
    BertModel.eval()


def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
    input_ids = input_ids[0][input_ids[0].nonzero()].transpose(0, 1)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)

    # We create a 3D attention mask from a 2D tensor mask.
    # Sizes are [batch_size, 1, 1, to_seq_length]
    # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    # this attention mask is more simple than the triangular masking of causal attention
    # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    embedding_output = self.embeddings(input_ids, token_type_ids)
    encoded_layers = self.encoder(embedding_output,
                                  extended_attention_mask,
                                  output_all_encoded_layers=output_all_encoded_layers)
    sequence_output = encoded_layers[-1]
    pooled_output = self.pooler(sequence_output)
    if not output_all_encoded_layers:
        encoded_layers = encoded_layers[-1]
    return encoded_layers[11][0, 0, :].unsqueeze_(0)


BertModel.forward = forward


def create_emb_layer(weights_matrix, non_trainable=False):
    weights_matrix = torch.from_numpy(weights_matrix)
    num_embeddings, embedding_dim = weights_matrix.size()

    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim
