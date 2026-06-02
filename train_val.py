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
from net.complex_MCFAUNet import MCFAUNet
from net.new_CUNet.new_CUNet import complex_UNet
from net.CMUNet import CMUNet
from utils import get_ids,split_ids,split_train_val,get_imgs_and_masks,batch,load_mat_imgs_masks
import time
import datetime
import pandas as pd

# mynet = 'CMAUNet'
# mynet = 'CMUNet'
mynet = 'CUNet'

N = '4'
# N = '6'
# N = '8'

# S = ''
S = 'S'

def train_net(net,
              epochs =100,
              batch_size =2,
              lr = 0.001,
              val_precent=0.15,
              save_cp=True,
              gpu=True,
              img_scale=1):

    dir_img = 'E:/tyz/脑部16线圈/C=40 N='+N+'/'+S+'train/'
    dir_mask = 'E:/tyz/脑部16线圈/'+S+'ref/'
    dir_checkpoint = 'E:/tyz/attention mri/conplex_CNN/complexcnn/checkpoints/'+S+mynet+'/'+N+'/'
    coilnum = 16   # multicoil number= 16

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

    # checkpoint = torch.load('D:/模型/zzcode/attention mri/complex_CNN/complexcnn/checkpoints/CMAUNet/6/CP44.pth')
    # net.load_state_dict(checkpoint['model_state_dict'])

    # net.load_state_dict(torch.load(dir_checkpoint+'CP99.pth'))
    # train_state = torch.load(dir_checkpoint+'State.pth')
    # start_epoch = train_state['epoch']
    # optimizer.load_state_dict(train_state['optimizer_state_dict'])
    #
    # for epoch in range(start_epoch+1, epochs):
    for epoch in range(epochs):

        # layer_names = []
        # gradient_norms = []
    #     begin_time = time()
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

            # 16 coils
            loss = criterion((masks_pred[:, 0:coilnum , :, :]), (true_masks[:, 0:coilnum , :, :])) + criterion((masks_pred[:, coilnum:coilnum*2, :, :]), (true_masks[:, coilnum:coilnum*2, :, :]))

            epoch_loss += loss.item()

            step+=1
            if step%10==0:
               # print(batch_size/N_train)
               print('{0:.4f} --- loss:{1:.8f}'.format(i*batch_size/N_train,loss.item()))
               # print(format(optimizer.state_dict()))

            optimizer.zero_grad()
            loss.backward()

            # for name, param in net.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.data.norm(2).item()
            #         layer_names.append(name)
            #         gradient_norms.append(grad_norm)

            optimizer.step()


        scheduler.step()

        # end_time = time()
        # run_time = end_time - begin_time
        # df = pd.DataFrame({
        #     'Layer Name': layer_names,
        #     'Gradient Norm': gradient_norms
        # })
        # excelname = f'gradient_data_epoch_{epoch+1}.xlsx'
        # df.to_excel(dir_checkpoint + excelname, index=False, engine='openpyxl')

        loss_dicegroup.append(epoch_loss/(i+1))
        lr_dicegroup.append((scheduler.get_last_lr()[0]))
        # print(format(scheduler.get_lr()[0]))
        print('epoch finished! loss:{}'.format(epoch_loss/(i+1)))
        # print('epoch time：', run_time)

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
            # checkpoint = {
            #     'model_state_dict': net.state_dict(),
            # }
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

    # fig = plt.figure()
    # plt.plot(Val_dicegroup, '*')
    # plt.plot(loss_dicegroup, '.')

def get_args():
    parser = OptionParser()

    # set default= 2 for demonstration, for example, set 100 for training.
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
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
    coilnum = 16   # multicoil number= 16
    net = None
    if mynet == 'CMAUNet':
        net = MCFAUNet(coilnum,coilnum).cuda()
    if mynet == 'CMUNet':
        net = CMUNet(coilnum,coilnum).cuda()
    if mynet == 'CUNet':
        net = complex_UNet(coilnum,coilnum).cuda()

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






