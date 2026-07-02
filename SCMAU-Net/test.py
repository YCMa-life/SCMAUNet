import torch
import scipy.io as sio
import numpy as np
from net.CMAUNet import CMAUNet
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dir_recon_img = "recon/C=40 N=4/"
dir_test_img = "../data/C=40 N=4/stest/"

pathDir = os.listdir(dir_test_img)

for testfilename in pathDir:

    images_under = sio.loadmat(dir_test_img+testfilename)
    images_under = images_under['imags']

    reconfilename=testfilename

    images_under = np.expand_dims(images_under, axis=0)
    images_under = torch.from_numpy(images_under)
    images_under = images_under.permute(0, 3, 1, 2)
    images_under = images_under.to(device)

    model = None
    model = CMAUNet(16,16).to(device)

    model_state_dict = torch.load("checkpoint/C=40 N=4/CP100.pth")
    model.load_state_dict(model_state_dict)
    predict = model(images_under).cpu()

    predict.cpu()
    predict = np.array(predict.detach().numpy(),dtype='float32')

    predict = np.squeeze(predict)
    predict = predict.transpose(1, 2, 0)
    sio.savemat(dir_recon_img+reconfilename,{'imags':predict})
