import glob
import os
import pickle

import numpy as np
import torch
from PIL import Image
from skimage import io
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from data_loader import SalObjDataset, RescaleT, ToTensorLab
from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB


# import torch.optim as optim

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')


def main():
    # --------- 1. get image path and name ---------
    model_name = 'u2netp'  # u2netp

    image_dir = '../train2014'
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = '../models/' + model_name + '.pth'

    img_name_list = glob.glob(image_dir + os.sep + '*')

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    net.load_state_dict(torch.load(model_dir))
    # if torch.cuda.is_available():
    #     net.cuda()
    net.eval()
    all_out = {}
    for i_test, data_test in tqdm(enumerate(test_salobj_dataloader)):
        sep_ = img_name_list[i_test].split(os.sep)[-1]
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        #
        # if torch.cuda.is_available():
        #     inputs_test = Variable(inputs_test.cuda())
        # else:
        inputs_test = Variable(inputs_test)
        d = net(inputs_test)

        pred = normPRED(d)
        all_out[sep_] = pred

    pickle.dump(all_out, open("../data/coco_train_u2net.pik", "wb"), protocol=2)


if __name__ == "__main__":
    main()
