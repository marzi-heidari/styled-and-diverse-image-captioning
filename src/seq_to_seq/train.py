import torch
import numpy as np
from torch import optim
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm

from seq_to_seq.utils import get_data, build_model, make_packpadded, epoch_to_save_path
from text_processing import tokenize_text, pad_text, untokenize

cuda = True
device = 0
BATCH_SIZE = 32



def shuffle_padded(tensor, order):
    for i in range(len(order) // 2):
        temp = tensor[i]
        tensor[i] = tensor[order[i]]
        tensor[order[i]] = temp


def train():
    input_text, input_rems_text = get_data(train=True)

    dec_idx_to_word, dec_word_to_idx, dec_tok_text, dec_bias = tokenize_text(input_text, lower_case=True, vsize=20000)
    dec_padded_text = pad_text(dec_tok_text)
    dec_vocab_size = len(dec_idx_to_word)

    enc_idx_to_word, enc_word_to_idx, enc_tok_text, _ = tokenize_text(input_rems_text)
    enc_padded_text = pad_text(enc_tok_text)
    enc_vocab_size = len(enc_idx_to_word)

    dec_text_tensor = torch.tensor(dec_padded_text, requires_grad=False)
    order = np.arange(enc_padded_text.shape[0])
    shuffle_padded(dec_text_tensor, order)
    if cuda:
        dec_text_tensor = dec_text_tensor.cuda(device=device)

    enc, dec = build_model(enc_vocab_size, dec_vocab_size, dec_bias=dec_bias)
    enc_optim, dec_optim, lossfunc = build_trainers(enc, dec)

    num_batches = enc_padded_text.shape[0] // BATCH_SIZE

    sm_loss = None
    enc.train()
    dec.train()
    for epoch in range(0, 13):
        print("Starting New Epoch: %d" % epoch)

        order = np.arange(enc_padded_text.shape[0])
        np.random.shuffle(order)
        enc_padded_text = enc_padded_text[order]
        dec_text_tensor.data = dec_text_tensor.data[order]

        for i in tqdm(range(num_batches)):
            s = i * BATCH_SIZE
            e = (i + 1) * BATCH_SIZE

            _, enc_pp, dec_pp, enc_lengths = make_packpadded(s, e, enc_padded_text, dec_text_tensor)

            enc.zero_grad()
            dec.zero_grad()

            hid = enc.initHidden(BATCH_SIZE)

            out_enc, hid_enc = enc.forward(enc_pp, hid, enc_lengths)

            hid_enc = torch.cat([hid_enc[0, :, :], hid_enc[1, :, :]], dim=1).unsqueeze(0)
            out_dec, hid_dec, attn = dec.forward(dec_pp[:, :-1], hid_enc, out_enc)

            out_perm = out_dec.permute(0, 2, 1)
            loss = lossfunc(out_perm, dec_pp[:, 1:])

            if sm_loss is None:
                sm_loss = loss.data
            else:
                sm_loss = sm_loss * 0.95 + 0.05 * loss.data

            loss.backward()
            clip_grad_value_(enc_optim.param_groups[0]['params'], 5.0)
            clip_grad_value_(dec_optim.param_groups[0]['params'], 5.0)
            enc_optim.step()
            dec_optim.step()

            # del loss
            if i % 100 == 0:
                print("Epoch: %.3f" % (i / float(num_batches) + epoch,), "Loss:", sm_loss)
                print("GEN:", untokenize(torch.argmax(out_dec, dim=2)[0, :], dec_idx_to_word))
                # print "GEN:", untokenize(torch.argmax(out_dec,dim=2)[1,:], dec_idx_to_word)
                print("GT:", untokenize(dec_pp[0, :], dec_idx_to_word))
                print("IN:", untokenize(enc_pp[0, :], enc_idx_to_word))

                print(torch.argmax(attn[0], dim=1))
                print("--------------")
        save_state(enc, dec, enc_optim, dec_optim, dec_idx_to_word, dec_word_to_idx, enc_idx_to_word, enc_word_to_idx,
                   epoch)


def save_state(enc, dec, enc_optim, dec_optim, dec_idx_to_word, dec_word_to_idx, enc_idx_to_word, enc_word_to_idx,
               epoch):
    state = {'enc': enc.state_dict(), 'dec': dec.state_dict(),
             'enc_optim': enc_optim.state_dict(), 'dec_optim': dec_optim.state_dict(),
             'dec_idx_to_word': dec_idx_to_word, 'dec_word_to_idx': dec_word_to_idx,
             'enc_idx_to_word': enc_idx_to_word, 'enc_word_to_idx': enc_word_to_idx}
    torch.save(state, epoch_to_save_path(epoch))


def build_trainers(enc, dec, loaded_state=None):
    learning_rate = 0.001
    lossfunc = nn.NLLLoss(ignore_index=0)

    enc_optim = optim.Adam(enc.parameters(), lr=learning_rate)
    dec_optim = optim.Adam(dec.parameters(), lr=learning_rate)
    if loaded_state is not None:
        enc_optim.load_state_dict(load_state['enc_optim'])
        dec_optim.load_state_dict(load_state['dec_optim'])
    return enc_optim, dec_optim, lossfunc


if __name__ == '__main__':
    train()
