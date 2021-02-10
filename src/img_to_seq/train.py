import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_

from img_to_seq.test import epoch_to_save_path
from img_to_seq.utils import build_model, get_data, BATCH_SIZE, device
from text_processing import tokenize_text, pad_text
from text_processing import untokenize


def train():
    feats, filenames, sents = get_data(train=True)
    dec_idx_to_word, dec_word_to_idx, dec_tok_text, dec_bias = tokenize_text(sents)
    dec_padded_text = pad_text(dec_tok_text)
    dec_vocab_size = len(dec_idx_to_word)
    enc, dec = build_model(dec_vocab_size, dec_bias)
    enc_optim, dec_optim, lossfunc = build_trainers(enc, dec)
    num_batches = feats.shape[0] // BATCH_SIZE

    feats_tensor = torch.tensor(feats, requires_grad=False)
    dec_text_tensor = torch.tensor(dec_padded_text, requires_grad=False)
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


def build_trainers(enc, dec, loaded_state=None):
    learning_rate = 0.001
    lossfunc = nn.NLLLoss(ignore_index=0)
    enc_optim = torch.optim.Adam(enc.parameters(), lr=learning_rate, weight_decay=1e-6)
    dec_optim = torch.optim.Adam(dec.parameters(), lr=learning_rate, weight_decay=1e-6)
    if loaded_state is not None:
        enc_optim.load_state_dict(load_state['enc_optim'])
        dec_optim.load_state_dict(load_state['dec_optim'])
    return enc_optim, dec_optim, lossfunc


def save_state(enc, dec, enc_optim, dec_optim, dec_idx_to_word, dec_word_to_idx, epoch):
    state = {'enc': enc.state_dict(), 'dec': dec.state_dict(),
             'enc_optim': enc_optim.state_dict(), 'dec_optim': dec_optim.state_dict(),
             'dec_idx_to_word': dec_idx_to_word, 'dec_word_to_idx': dec_word_to_idx}
    torch.save(state, epoch_to_save_path(epoch))


if __name__ == '__main__':
    train()
