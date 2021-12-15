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
from utils import AverageMeter
import pickle

import PIL
from PIL import Image as im
from weighted_levenshtein import lev, osa, dam_lev

pt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract" # location of pytesseract --- change as needed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir): os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = FSRCNN(scale_factor=args.scale).to(device)
    def tensor_to_image(tensor):
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    optimizer = optim.Adam([{'params': model.first_part.parameters()}, {'params': model.mid_part.parameters()}, {'params': model.last_part.parameters(), 'lr': args.lr * 0.1}], lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    eval_counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('Epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data[0].to(device), data[1]
                preds = model(inputs)
                tensor_to_image(preds[0].detach().numpy().reshape((20, 20))).save("1.jpg")
                loss_f = nn.MSELoss()
                loss = loss_f(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        model.eval()

    for data in eval_dataloader:
        inputs, labels = data[0].to(device), data[1]
        with torch.no_grad(): preds = model(inputs).clamp(0.0, 255.0)
        tensor_to_image(preds[0].reshape((20, 20))).save(args.outputs_dir + str(eval_counter) + ".jpg")
        eval_counter+=1
# 0.000370268038*1000*200 \approx 37%