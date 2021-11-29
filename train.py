import argparse
import os
import copy
import math
import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import pytesseract as pt
from torchvision import transforms
import matplotlib.pyplot as plt

from models import FSRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr

from PIL import Image as im
from weighted_levenshtein import lev, osa, dam_lev

pt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract"
#python train.py --train-file "dataset/91-image_x4.h5" --eval-file "dataset/Set5_x4.h5" --outputs-dir "output" --scale 4 --batch-size 16 --num-epochs 20 --num-workers 8

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--weights-file', type=str)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = FSRCNN(scale_factor=args.scale).to(device)

    insc, delc, subc, transc = np.ones(128, dtype=np.float64), np.ones(128, dtype=np.float64), np.ones((128, 128), dtype=np.float64), np.ones((128, 128), dtype=np.float64) # weighted Levenshtein cost matrices
    epsilon = 1
    def wLevdist(tensor1, tensor2):
        str1, str2 = "".join(pt.image_to_string(transforms.ToPILImage()(tensor1).convert("RGB"), config="-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVQXYZ --psm 6").split()), "".join(pt.image_to_string(transforms.ToPILImage()(tensor2).convert("RGB"), config="-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVQXYZ --psm 6").split())
        return epsilon*2*dam_lev(str1, str2, insert_costs = insc, delete_costs = delc, substitute_costs = subc, transpose_costs = transc)/max(1, len(str1) + len(str2)) # average weighted Levenshtein distance based on average number of characters in both strings

    def loss_fn(output, ground):
        t = torch.Tensor(torch.sum(torch.Tensor([wLevdist(output[a], ground[a]) for a in range(len(output))]))/len(output))
        t.requires_grad = True
        return 1*t
    criterion = loss_fn
    #nn.MSELoss() # loss function

    optimizer = optim.Adam([
        {'params': model.first_part.parameters()},
        {'params': model.mid_part.parameters()},
        {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    eval_dataset = EvalDataset(args.eval_file) # make a training dataset out of OCR images with ground truth, then adjust Levenshtein distance function accordingly
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = math.inf

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                #print(inputs)
                labels = labels.to(device)
                #print(labels)

                preds = model(inputs) # takes in 16 training samples and produces 16 outputs, which are then used for learning; each image is (24, 24), which is very small
                """for a in range(len(preds.detach().numpy())):
                    p = 255*np.abs(preds.detach().numpy()[a].reshape((24, 24)))/np.max(np.abs(preds.detach().numpy()[a].reshape((24, 24))))
                    #print(p)
                    im.fromarray(p).convert("L").save(str(epoch) + str(a) + '.png')
                """
                #print(preds.detach().numpy().shape, " fsf asd as", labels.numpy().shape)
                loss = criterion(preds, labels) # loss is calculated between the output image and the label, which is a scaled-down version of the image that is upscaled. This should be replaced with a loss function that applies MSE to the per-character weighted Levenshtein distance between the ground truth and output text - perhaps this could be done by running OCR directly on each prediction before feeding it into the loss function, which will be very basic in nature
                # Modify FSRCNN by modifying loss function --- figure out how loss is calculatedm then feed in 
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                best_loss = min(best_loss, epoch_losses.avg)
                t.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
    print('best avg loss: {}'.format(best_loss))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))