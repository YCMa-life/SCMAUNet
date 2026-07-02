import torch
import numpy as np
import torch.nn as nn
from utils import batch

criterion= nn.MSELoss()
def eval_net(net,val,batch_size,gpu):

    with torch.no_grad():
        net.training = False
        epoch_loss =0
        i=0
        coilnum = 16

        for i, b in enumerate(batch(val, batch_size)):

            imgs = np.array([m[0] for m in b])
            true_masks = np.array([m[1] for m in b])

            imgs = torch.from_numpy(imgs).float()
            imgs = imgs.permute(0, 3, 1, 2)

            true_masks = torch.from_numpy(true_masks).float()
            true_masks = true_masks.permute(0, 3, 1, 2)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)

            loss = criterion((masks_pred[:, 0:coilnum, :, :]), (true_masks[:, 0:coilnum, :, :])) + criterion(
                (masks_pred[:, coilnum:coilnum * 2, :, :]), (true_masks[:, coilnum:coilnum * 2, :, :]))

            epoch_loss += loss.item()

        return epoch_loss / (i+1)