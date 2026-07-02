# Reference link: https://github.com/ZhyJin/SCUNET

import sys
import os
from optparse import OptionParser
import numpy as np
import scipy.io as sio
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from eval import eval_net
from net.CMAUNet import CMAUNet
from utils import get_ids,split_ids,split_train_val,get_imgs_and_masks,batch,load_mat_imgs_masks

def train_net(net,
              epochs =100,
              batch_size =2,
              lr = 0.001,
              val_precent=0.15,
              save_cp=True,
              gpu=True,
              img_scale=1):

    dir_img = "../data/C=40 N=4/strain/"
    dir_mask = "../data/sref_train/"
    dir_checkpoint = "checkpoint/C=40 N=4/"

    coilnum = 16

    print("""
    start train:
    epochs:{}
    batch_size:{}
    learning_rate:{}
    checkpoint:{}
    cuda:{}
    """.format(epochs,batch_size,lr,str(save_cp),str(gpu)))

    optimizer = optim.Adam(net.parameters(),lr=lr,betas=(0.9, 0.999))

    criterion = nn.MSELoss()

    scheduler=optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    Val_dicegroup=[]
    loss_dicegroup = []
    lr_dicegroup=[]

    for epoch in range(epochs):

        ids = get_ids(dir_img)
        ids = split_ids(ids)
        iddataset = split_train_val(ids, val_precent)
        N_train = len(iddataset['train'])

        print("""
        start epoch:{}/{}
        train_size:{}
        val_size:{}
        """.format((epoch+1),epochs,len(iddataset['train']),len(iddataset['val'])))

        net.train()

        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask,img_scale)
        val = get_imgs_and_masks(iddataset['val'],dir_img,dir_mask,img_scale)

        epoch_loss=0
        step = 0
        i=0
        for i,b in enumerate(batch(train,batch_size)):
            imgs = np.array([m[0]for m in b])
            true_masks = np.array([m[1]for m in b])

            imgs = torch.from_numpy(imgs)
            imgs= imgs.permute(0, 3, 1, 2)

            true_masks = torch.from_numpy(true_masks)
            true_masks = true_masks.permute(0, 3, 1, 2)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)

            loss = criterion((masks_pred[:, 0:coilnum , :, :]), (true_masks[:, 0:coilnum , :, :])) \
                   + criterion((masks_pred[:, coilnum:coilnum*2, :, :]), (true_masks[:, coilnum:coilnum*2, :, :]))

            epoch_loss += loss.item()

            step+=1
            if step%10==0:
               print('{0:.4f} --- loss:{1:.8f}'.format(i * batch_size/N_train,loss.item()))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        scheduler.step()

        loss_dicegroup.append(epoch_loss/(i+1))
        lr_dicegroup.append((scheduler.get_last_lr()[0]))

        print('epoch finished! loss:{}'.format(epoch_loss/(i+1)))

        if epoch >= epochs - 10:
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()

            if 1:
                val_dice = eval_net(net, val, batch_size, gpu)
                Val_dicegroup.append(val_dice)
                print('validation dice coeff:{}'.format(val_dice))

        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

        if save_cp:

            train_state = {
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            torch.save(train_state,
                       dir_checkpoint + 'State.pth')
            print('checkpoint{}saved!'.format(epoch+1))

            sio.savemat(dir_checkpoint + 'Evalfile.mat', {'Eval': Val_dicegroup})
            sio.savemat(dir_checkpoint + 'Lossfile.mat', {'Loss': loss_dicegroup})
            sio.savemat(dir_checkpoint + 'Lrfile.mat', {'Lr': lr_dicegroup})


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int',
                      help='number of epochs')

    parser.add_option('-b', '--batch-size', dest='batchsize', default=2,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                      type='float', help='learning_rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=True, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=1, help='downscaling factor of the images')
    (options,args) = parser.parse_args()
    return options

if __name__ =='__main__':

    args = get_args()
    coilnum = 16
    net = None
    net = CMAUNet(coilnum,coilnum).cuda()

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)

    except KeyboardInterrupt:
        torch.save(net.state_dict(),'INTERRUPTED.pth')
        print('saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)






